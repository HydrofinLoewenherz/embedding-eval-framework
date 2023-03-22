import itertools

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
from tqdm.notebook import tqdm
import networkx as nx
import torch
from torch import nn, FloatTensor
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryAveragePrecision, BinaryConfusionMatrix, BinaryF1Score
from typing import List, Union, Tuple, Any

from src.args import Args
from src.graph import NodeDataPairs, random_geometric_graph, subgraph


class DatasetBuilder:
    def __init__(self, graph: nx.Graph, batch_size: int, device):
        self.graph = graph
        self.batch_size = batch_size

        self.n_nodes = len(self.graph.nodes(data=False))
        self.n_edges = len(self.graph.edges(data=False))

        self.node_feature_pairs: NodeDataPairs = set(itertools.combinations(self.graph.nodes.data("feature"), 2))
        values, labels = zip(*[
            (
                [*u_f, *v_f],
                1 if graph.has_edge(u, v) else 0
            )
            for ((u, u_f), (v, v_f)) in self.node_feature_pairs
        ])
        self.size = len(values)
        self.n_non_edges = self.size - self.n_edges
        self.ds_values = torch.FloatTensor(values).to(device)
        self.ds_labels = torch.IntTensor(labels).to(device)

        self.dataset = TensorDataset(self.ds_values, self.ds_labels)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int, layer_size: int):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, 1)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


def result_figure(
        graph: nx.Graph,
        dataset: DatasetBuilder,
        preds: List[float],
        threshold: float = 0.5,
        node_colors: Union[List[str], None] = None
) -> any:
    fig, ax = plt.subplots(1, 2)
    node_size = 500 / dataset.n_nodes

    # add original graph (with provided node colors)
    if node_colors is None:
        node_colors = "blue"

    ax[0].set_axis_off()
    ax[0].set_aspect('equal')
    ax[0].set_title("original graph")
    nx.draw_networkx(
        graph,
        pos=graph.nodes.data("pos"),
        ax=ax[0],
        node_size=node_size,
        with_labels=False,
        labels={},
        node_color=node_colors
    )

    # add predict graph
    pred_graph = nx.Graph()
    pred_graph.add_nodes_from(dataset.graph.nodes(data=True))
    pred_graph.add_edges_from([
        (u, v, {"pred": preds[i]})
        for i, ((u, _), (v, _)) in enumerate(dataset.node_feature_pairs)
        if preds[i] > threshold
    ])
    colormap = sns.color_palette("flare", as_cmap=True)

    ax[1].set_axis_off()
    ax[1].set_aspect('equal')
    ax[1].set_title("reconstructed graph")
    nx.draw_networkx(
        pred_graph,
        pos=graph.nodes.data("pos"),
        ax=ax[1],
        node_size=node_size,
        with_labels=False,
        labels={},
        edge_color=[pred for (_, _, pred) in pred_graph.edges(data="pred")],
        edge_cmap=colormap
    )

    # add color bar for predictions
    cax = fig.add_axes([ax[1].get_position().x1 + 0.01, ax[1].get_position().y0, 0.02, ax[1].get_position().height])
    fig.colorbar(mpl.cm.ScalarMappable(cmap=colormap), cax=cax, label="confidence")

    return fig


class Evaluator:
    def __init__(self, graph: nx.Graph, dim: int, args: Args, writer_log_dir: str, device):
        with tqdm(total=10, desc="building evaluator") as pbar:
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.device = device
            self.args = args
            self.dim = dim
            self.writer = SummaryWriter(writer_log_dir)
            self.net = NeuralNetwork(
                input_size=dim*2,
                layer_size=args.layer_size
            ).to(device)
            pbar.update(1)
            self.graph = graph
            pbar.update(1)
            self.whole_dataset = DatasetBuilder(
                graph=self.graph,
                batch_size=self.args.batch_size,
                device=self.device
            )
            pbar.update(1)

            # split graph into train and test
            self.test_graph, test_sampled = subgraph(
                graph=self.graph,
                size=int(self.whole_dataset.n_nodes * self.args.test_split),
                # try to keep split data strongly connected
                alpha=self.args.epoch_graph_alpha,
                boredom_pth=0.9
            )
            pbar.update(1)
            self.valid_graph, valid_sampled = subgraph(
                graph=self.graph,
                size=int(self.whole_dataset.n_nodes * self.args.valid_split),
                # try to keep split data strongly connected
                alpha=self.args.epoch_graph_alpha,
                boredom_pth=0.9,
                ignore=test_sampled
            )
            pbar.update(1)
            # give the rest to the train graph
            valid_test_nodes = [*valid_sampled, *test_sampled]
            self.train_graph = nx.subgraph(self.graph, [
                n
                for n in self.graph.nodes(data=False)
                if n not in valid_test_nodes
            ])
            pbar.update(1)
            self.test_dataset = DatasetBuilder(
                graph=self.test_graph,
                batch_size=self.args.batch_size,
                device=self.device
            )
            pbar.update(1)

            self.valid_dataset = DatasetBuilder(
                graph=self.valid_graph,
                batch_size=self.args.batch_size,
                device=self.device
            )
            pbar.update(1)
            self.train_dataset = DatasetBuilder(
                graph=self.train_graph,
                batch_size=self.args.batch_size,
                device=self.device
            )
            pbar.update(1)

            self.ap_score_fn = BinaryAveragePrecision().to(self.device)
            self.f1_score_fn = BinaryF1Score().to(self.device)
            self.bcm_fn = BinaryConfusionMatrix().to(self.device)
            pbar.update(1)

    def __fit(self, dataset: DatasetBuilder, optimizer) -> Tuple[float, FloatTensor]:
        self.net.train()
        losses = []
        preds = []
        for i_batch, (x_train, y_train) in enumerate(dataset.dataloader):
            optimizer.zero_grad()
            y_pred = self.net(x_train)  # if the loss_fn is with logits, don't use sigmoid
            loss = self.loss_fn(y_pred, y_train.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            preds.extend(y_pred.unsqueeze(1))
        preds = torch.sigmoid(torch.FloatTensor(preds).to(self.device))
        return float(np.mean(losses)), preds

    def __score(self, dataset: DatasetBuilder) -> Tuple[float, FloatTensor]:
        self.net.eval()
        losses = []
        preds = []
        for i_batch, (x_train, y_train) in enumerate(dataset.dataloader):
            y_pred = self.net(x_train)  # if the loss_fn is with logits, don't use sigmoid
            loss = self.loss_fn(y_pred, y_train.unsqueeze(1).float())
            losses.append(loss.item())
            preds.extend(y_pred.unsqueeze(1))
        preds = torch.sigmoid(torch.FloatTensor(preds).to(self.device))
        return float(np.mean(losses)), preds

    def train(self, optimizer, save_fig: bool = True):
        self.writer.add_text("args", self.args.__repr__())

        # early stopping
        best_score = 0
        patience_counter = 0
        patience = 20

        with tqdm(desc="training model...") as pbar:
            for epoch in range(self.args.epochs):
                pbar.set_description(f"epoch {epoch + 1}")

                # generate graph and dataset for epoch (and track some metrics)
                epoch_graph, _ = subgraph(
                    size=self.args.epoch_graph_size,
                    graph=self.train_graph,
                    alpha=self.args.epoch_graph_alpha,
                    boredom_pth=self.args.epoch_graph_boredom_pth
                )
                epoch_dataset = DatasetBuilder(
                    graph=epoch_graph,
                    batch_size=self.args.batch_size,
                    device=self.device
                )
                self.writer.add_scalar('subgraph_edges', epoch_dataset.n_edges / epoch_dataset.size, epoch)
                pbar.update(1)

                # train epoch
                fit_loss, _ = self.__fit(epoch_dataset, optimizer)
                self.writer.add_scalar('fit_loss', fit_loss, epoch)
                pbar.update(1)

                # evaluate on epoch dataset
                train_loss, train_preds = self.__score(epoch_dataset)
                ap_score = self.ap_score_fn(train_preds, epoch_dataset.ds_labels)
                f1_score = self.f1_score_fn(train_preds, epoch_dataset.ds_labels)
                self.writer.add_scalar('train_loss', train_loss, epoch)
                self.writer.add_scalar('train_precision', ap_score, epoch)
                self.writer.add_scalar('train_f1', f1_score, epoch)
                threshold = f1_score.item()
                pbar.update(1)

                # evaluate on valid dataset for early stopping
                valid_loss, valid_preds = self.__score(self.valid_dataset)
                valid_ap = self.ap_score_fn(valid_preds, self.valid_dataset.ds_labels)
                self.writer.add_scalar('valid_loss', valid_loss, epoch)
                self.writer.add_scalar('valid_precision', valid_ap, epoch)
                pbar.update(1)

                # stop early if no improvement for last epochs
                if valid_ap > best_score:
                    best_score = valid_ap
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter > patience and self.args.early_stopping:
                        break

                # save subgraph periodically
                if save_fig and epoch % 10 == 0:
                    node_colors = list(dict(sorted({
                        **{n: 'green' for n in list(self.train_graph.nodes)},
                        **{n: 'red' for n in list(epoch_graph.nodes)}
                    }.items())).values())
                    f = result_figure(
                        graph=self.train_graph,
                        dataset=epoch_dataset,
                        preds=train_preds.cpu().numpy(),
                        threshold=threshold,
                        node_colors=node_colors
                    )
                    self.writer.add_figure('epoch_graph', f, epoch)
                pbar.update(1)

    def test(self, epoch: Union[int, None] = None, save_fig: bool = True) -> Tuple[Any, Any, Any]:
        with tqdm(total=5, desc="testing model") as pbar:
            self.net.eval()

            # evaluate test dataset (in batches)
            test_loss, test_preds = self.__score(self.test_dataset)
            pbar.update(1)

            # ap score
            test_ap = self.ap_score_fn(test_preds, self.test_dataset.ds_labels)
            self.writer.add_scalar('test_precision', test_ap, epoch)
            pbar.update(1)

            # confusion matrix
            test_conf = self.bcm_fn(test_preds, self.test_dataset.ds_labels)
            self.writer.add_text(
                "test_confusion",
                f"""
                    non-edge | right: {test_conf[0][0]} / wrong: {test_conf[0][1]} / all: {self.test_dataset.n_non_edges}
                    edge | right: {test_conf[1][1]} / wrong: {test_conf[1][0]} / all: {self.test_dataset.n_edges}
                """,
                epoch
            )
            pbar.update(1)

            # f1 score (also for threshold in reconstruction)
            test_f1 = self.f1_score_fn(test_preds, self.test_dataset.ds_labels)
            self.writer.add_scalar('test_f1', test_f1, epoch)
            threshold = test_f1.item()
            pbar.update(1)

            # color train dataset in green and test dataset in red
            if save_fig:
                node_colors = list(dict(sorted({
                    **{n: 'green' for n in list(self.graph.nodes)},
                    **{n: 'red' for n in list(self.test_graph.nodes)}
                }.items())).values())
                f = result_figure(
                    graph=self.graph,
                    dataset=self.test_dataset,
                    preds=test_preds.cpu().numpy(),
                    threshold=threshold,
                    node_colors=node_colors
                )
                self.writer.add_figure('test_graph', f, epoch)
            pbar.update(1)

            return test_loss, test_ap.item(), test_f1.item()

    def eval(self, epoch: Union[int, None] = None, save_fig: bool = True):
        with tqdm(total=4, desc="evaluating model") as pbar:
            self.net.eval()

            # evaluate whole dataset (in batches)
            eval_loss, eval_preds = self.__score(self.whole_dataset)
            self.writer.add_scalar('eval_loss', eval_loss, epoch)
            pbar.update(1)

            # ap score
            eval_ap = self.ap_score_fn(eval_preds, self.whole_dataset.ds_labels)
            self.writer.add_scalar('eval_precision', eval_ap, epoch)
            pbar.update(1)

            # f1 score for threshold in reconstruction
            eval_f1 = self.f1_score_fn(eval_preds, self.whole_dataset.ds_labels)
            self.writer.add_scalar('eval_f1', eval_f1, epoch)
            threshold = eval_f1.item()
            pbar.update(1)

            if save_fig:
                f = result_figure(
                    graph=self.graph,
                    dataset=self.whole_dataset,
                    preds=eval_preds.cpu().numpy(),
                    threshold=threshold
                )
                self.writer.add_figure('eval_graph', f, epoch)
            pbar.update(1)
