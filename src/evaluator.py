import itertools

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
from tqdm.notebook import tqdm
import networkx as nx
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryAveragePrecision, BinaryConfusionMatrix
from typing import List, Union

from src.args import Args
from src.graph import NodeDataPairs, random_geometric_graph, subgraph


class DatasetBuilder:
    def __init__(self, graph: nx.Graph, batch_size: int, shuffle: bool, device):
        self.graph = graph
        self.batch_size = batch_size

        self.n_nodes = len(self.graph.nodes(data=False))
        self.n_edges = len(self.graph.edges(data=False))

        self.node_pos_pairs: NodeDataPairs = set(itertools.combinations(self.graph.nodes.data("pos"), 2))
        values, labels = zip(*[
            (
                [*u_p, *v_p],
                1 if graph.has_edge(u, v) else 0
            )
            for ((u, u_p), (v, v_p)) in self.node_pos_pairs
        ])
        self.size = len(values)
        self.n_non_edges = self.size - self.n_edges
        self.ds_values = torch.FloatTensor(values).to(device)
        self.ds_labels = torch.IntTensor(labels).to(device)

        self.dataset = TensorDataset(self.ds_values, self.ds_labels)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)


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


def result_figure(graph: nx.Graph, dataset: DatasetBuilder, preds: List[float],
                  threshold: float = 0.5, node_colors: Union[List[str], None] = None) -> any:
    fig, ax = plt.subplots(1, 2)
    node_size = 2000 / dataset.n_nodes

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
    colors_filtered = np.array([
        preds[i]
        for i, _ in enumerate(dataset.node_pos_pairs)
        if preds[i] > threshold
    ])
    colormap = sns.color_palette("flare", as_cmap=True)
    pred_graph = nx.Graph()
    pred_graph.add_nodes_from(graph.nodes(data=True))
    pred_graph.add_edges_from([
        (u, v)
        for i, ((u, _), (v, _)) in enumerate(dataset.node_pos_pairs)
        if preds[i] > threshold
    ])

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
        edge_color=colors_filtered,
        edge_cmap=colormap
    )

    # add color bar for predictions
    cax = fig.add_axes([ax[1].get_position().x1 + 0.01, ax[1].get_position().y0, 0.02, ax[1].get_position().height])
    fig.colorbar(mpl.cm.ScalarMappable(cmap=colormap), cax=cax, label="confidence")

    return fig


class Evaluator:
    def __init__(self, args: Args, writer_log_dir: str, device):
        self.device = device
        self.args = args
        self.writer = SummaryWriter(writer_log_dir)
        self.net = NeuralNetwork(
            input_size=4,
            layer_size=args.layer_size
        ).to(device)
        self.graph = random_geometric_graph(
            size=args.graph_size,
            radius=args.rg_radius
        )

        # split graph into train and test
        self.test_graph = subgraph(
            graph=self.graph,
            size=int(len(self.graph.nodes(data=False)) * 0.3),
            # try to keep split data strongly connected
            alpha=1.0,
            boredom_pth=0.9
        )
        self.train_graph = nx.subgraph(self.graph, [
            n
            for n in self.graph.nodes(data=False)
            if not self.test_graph.has_node(n)
        ])

        self.whole_dataset = DatasetBuilder(
            graph=self.graph,
            batch_size=self.args.batch_size,
            shuffle=True,
            device=self.device
        )
        self.test_dataset = DatasetBuilder(
            graph=self.test_graph,
            batch_size=self.args.batch_size,
            shuffle=True,
            device=self.device
        )
        self.train_dataset = DatasetBuilder(
            graph=self.train_graph,
            batch_size=self.args.batch_size,
            shuffle=True,
            device=self.device
        )

    def train(self, loss_fn, optimizer):
        ap_curve = BinaryAveragePrecision()
        self.writer.add_text("args", self.args.__repr__())

        with tqdm(total=self.args.epochs * 4, desc="training model...") as pbar:
            for epoch in range(self.args.epochs):
                pbar.set_description(f"epoch {epoch + 1}")

                # generate graph and dataset for epoch (and track some metrics)
                epoch_graph = subgraph(
                    size=self.args.subsample_size,
                    graph=self.train_graph,
                    alpha=0.1,
                    boredom_pth=0.3
                )
                epoch_dataset = DatasetBuilder(
                    graph=epoch_graph,
                    batch_size=self.args.batch_size,
                    shuffle=True,
                    device=self.device
                )
                self.writer.add_scalar('subgraph_edges', epoch_dataset.n_edges / epoch_dataset.size, epoch)
                pbar.update(1)

                # train epoch (in batches)
                self.net.train()
                train_losses = []
                for i_batch, (x_train, y_train) in enumerate(epoch_dataset.dataloader):
                    optimizer.zero_grad()
                    y_pred = self.net(x_train)
                    # TODO run sigmoid?!
                    loss = loss_fn(y_pred, y_train.unsqueeze(1).float())
                    loss.backward()  # is this right?!
                    optimizer.step()
                    train_losses.append(loss.item())
                mean_train_loss = np.mean(train_losses)
                self.writer.add_scalar('mean_train_loss', mean_train_loss, epoch)
                pbar.update(1)

                # evaluate epoch after train (in batches)
                self.net.eval()
                eval_preds = []
                for i_batch, (x_train, y_train) in enumerate(epoch_dataset.dataloader):
                    y_pred = self.net(x_train)
                    eval_preds.extend(y_pred.unsqueeze(1))
                eval_preds = torch.FloatTensor(eval_preds).to(self.device)
                pbar.update(1)
                # TODO run sigmoid?!

                # learning rate TODO

                # ap score
                ap_score = ap_curve(eval_preds, epoch_dataset.ds_labels)
                self.writer.add_scalar('average_train_precision', ap_score, epoch)

                # f1 score (also for threshold in reconstruction) TODO
                threshold = 0.5

                # run further evaluations
                if self.args.eval_epochs and epoch % 10 == 0:
                    # save subgraph
                    node_colors = list(dict(sorted({
                        **{n: 'green' for n in list(self.train_graph.nodes)},
                        **{n: 'red' for n in list(epoch_graph.nodes)}
                    }.items())).values())
                    f = result_figure(
                        graph=self.train_graph,
                        dataset=epoch_dataset,
                        preds=eval_preds.cpu(),
                        threshold=threshold,
                        node_colors=node_colors
                    )
                    self.writer.add_figure('epoch_graph', f, epoch)

                    # eval on test dataset
                    self.test(epoch)
                pbar.update(1)

    def test(self, epoch: Union[int, None] = None):
        self.net.eval()

        # evaluate test dataset (in batches)
        eval_preds = []
        for i_batch, (x_train, y_train) in enumerate(self.test_dataset.dataloader):
            y_pred = self.net(x_train)
            eval_preds.extend(y_pred.unsqueeze(1))
        eval_preds = torch.FloatTensor(eval_preds).to(self.device)
        # TODO run sigmoid?!

        # ap score
        ap_curve = BinaryAveragePrecision()
        ap_score = ap_curve(eval_preds, self.test_dataset.ds_labels)
        self.writer.add_scalar('average_test_precision', ap_score, epoch)

        # confusion matrix
        bcm = BinaryConfusionMatrix(validate_args=True).to(self.device)
        conf = bcm(eval_preds, self.test_dataset.ds_labels)
        self.writer.add_text(
            "graph_confusion",
            f"""
                non-edge | right: {conf[0][0]} / wrong: {conf[0][1]} / all: {self.test_dataset.n_non_edges}
                edge | right: {conf[1][1]} / wrong: {conf[1][0]} / all: {self.test_dataset.n_edges}
            """,
            epoch
        )

        # f1 score (also for threshold in reconstruction) TODO
        threshold = 0.5

        f = result_figure(
            graph=self.test_graph,
            dataset=self.test_dataset,
            preds=eval_preds.cpu(),
            threshold=threshold
        )
        self.writer.add_figure('test_graph', f, epoch)

    def eval(self, epoch: Union[int, None] = None):
        # evaluate whole dataset (in batches)
        whole_preds = []
        for i_batch, (x_train, y_train) in enumerate(self.whole_dataset.dataloader):
            y_pred = self.net(x_train)
            whole_preds.extend(y_pred.unsqueeze(1))
        whole_preds = torch.FloatTensor(whole_preds).to(self.device)
        # TODO run sigmoid?!

        # f1 score for threshold in reconstruction TODO
        threshold = 0.5

        # color train dataset in green and test dataset in red

        node_colors = list(dict(sorted({
           **{n: 'green' for n in list(self.graph.nodes)},
           **{n: 'red' for n in list(self.test_graph.nodes)}
        }.items())).values())
        f = result_figure(
            graph=self.graph,
            dataset=self.whole_dataset,
            preds=whole_preds.cpu(),
            threshold=threshold,
            node_colors=node_colors
        )
        self.writer.add_figure('whole_graph', f, epoch)
