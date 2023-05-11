import itertools
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import networkx as nx
import torch
from torch import nn, Tensor
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryAveragePrecision, BinaryConfusionMatrix, BinaryF1Score, BinaryPrecisionRecallCurve
from typing import Union, Tuple

from src.pytorchtools import EarlyStopping

from src.args import Args
import src.graphs as graphs
import src.subgraphs as subgraphs
import src.graph_visualization as visualization


class DatasetBuilder:
    def __init__(self, graph: nx.Graph, batch_size: int, device):
        self.graph = graph
        self.batch_size = batch_size

        self.n_nodes = len(self.graph.nodes(data=False))
        self.n_edges = len(self.graph.edges(data=False))

        # build dataset
        # expect the nodes to be in random order, as such we effectively choose one of the two f_uv and f_vu at random
        self.node_feature_pairs = list(itertools.combinations(list(self.graph.nodes(data="feature")), 2))
        values, labels = zip(*[
            (
                # combined feature vector
                [*u_f, *v_f],
                # label
                1 if graph.has_edge(u, v) else 0
            )
            for ((u, u_f), (v, v_f)) in self.node_feature_pairs
        ])
        self.dim = len(values[0])
        self.size = len(values)
        self.n_non_edges = self.size - self.n_edges
        self.ds_values = torch.FloatTensor(values).to(device)
        self.ds_labels = torch.IntTensor(labels).to(device)

        self.dataset = TensorDataset(self.ds_values, self.ds_labels)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


class Evaluator:
    def __init__(self, graph: nx.Graph, args: Args, writer_log_dir: str, device):
        self.device = device
        self.args = args
        self.writer = SummaryWriter(writer_log_dir)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.pr_curve_fn = BinaryPrecisionRecallCurve().to(self.device)
        self.ap_score_fn = BinaryAveragePrecision().to(self.device)
        self.f1_score_fn = BinaryF1Score().to(self.device)
        self.bcm_fn = BinaryConfusionMatrix().to(self.device)

        # apply optional transformations on the graph
        self.graph = graph
        if args.std_dataset:
            self.graph = graphs.standardize_graph(self.graph)
        if args.sort_dataset:
            self.graph = graphs.sorted_graph(self.graph)

        # build dataset for whole graph
        self.whole_dataset = DatasetBuilder(
            graph=self.graph,
            batch_size=self.args.batch_size,
            device=self.device,
        )
        self.dim = self.whole_dataset.dim

        # build model
        self.net = NeuralNetwork(
            input_size=self.dim,
        ).to(device)

        # split graph and build datasets
        dataset_split = list(map(int, self.args.dataset_split.split("-")))
        valid_split = dataset_split[1] / np.sum(dataset_split)
        test_split = dataset_split[2] / np.sum(dataset_split)
        unsampled_nodes = set(self.graph.nodes(data=False))
        split_alg = self.args.subgraph_alg
        # the split algorithm has to subsample ('none' does not work here)
        if split_alg == "none":
            split_alg = "rjs"
        self.test_graph = subgraphs.sub_of(
            alg=split_alg,
            graph=self.graph,
            size=int(self.whole_dataset.n_nodes * test_split),
            alpha=self.args.subgraph_alpha,
            beta=0.9  # try to keep split data strongly connected
        )
        unsampled_nodes -= set(self.test_graph.nodes(data=False))
        self.valid_graph = subgraphs.sub_of(
            alg=split_alg,
            graph=nx.subgraph(self.graph, unsampled_nodes),
            size=int(self.whole_dataset.n_nodes * valid_split),
            alpha=self.args.subgraph_alpha,
            beta=0.9  # try to keep split data strongly connected
        )
        unsampled_nodes -= set(self.valid_graph.nodes(data=False))
        self.train_graph = nx.subgraph(self.graph, unsampled_nodes)

        self.test_dataset = DatasetBuilder(
            graph=self.test_graph,
            batch_size=self.args.batch_size,
            device=self.device,
        )
        self.valid_dataset = DatasetBuilder(
            graph=self.valid_graph,
            batch_size=self.args.batch_size,
            device=self.device,
        )
        self.train_dataset = DatasetBuilder(
            graph=self.train_graph,
            batch_size=self.args.batch_size,
            device=self.device,
        )

    def fit(self, dataloader: DataLoader, optimizer) -> Tuple[float, Tensor]:
        self.net.train()
        losses = []
        preds = []
        for i_batch, (x_train, y_train) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred = self.net(x_train)  # if the loss_fn is with logits, don't use sigmoid
            loss = self.loss_fn(y_pred, y_train.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            preds.extend(y_pred.unsqueeze(1))
        preds = torch.sigmoid(torch.FloatTensor(preds).to(self.device))
        return float(np.mean(losses)), preds

    def score(self, dataloader: DataLoader) -> Tuple[float, Tensor]:
        self.net.eval()
        losses = []
        preds = []
        for i_batch, (x_train, y_train) in enumerate(dataloader):
            y_pred = self.net(x_train)  # if the loss_fn is with logits, don't use sigmoid
            loss = self.loss_fn(y_pred, y_train.unsqueeze(1).float())
            losses.append(loss.item())
            preds.extend(y_pred.unsqueeze(1))
        preds = torch.sigmoid(torch.FloatTensor(preds).to(self.device))
        return float(np.mean(losses)), preds

    def train(self, optimizer, pbar: bool = True, track: bool = False) -> pd.DataFrame:
        early_stopper = EarlyStopping(patience=20, path="./out/model.pt")
        track_record = pd.DataFrame({})
        start_time = time.perf_counter()

        for epoch in (tqdm(range(self.args.epochs)) if pbar else range(self.args.epochs)):
            # generate graph and dataset for epoch
            epoch_graph = subgraphs.sub_of(
                alg=self.args.subgraph_alg,
                graph=self.train_graph,
                size=self.args.subgraph_size,
                alpha=self.args.subgraph_alpha,
                beta=self.args.subgraph_beta,
            )
            epoch_dataset = DatasetBuilder(
                graph=epoch_graph,
                batch_size=self.args.batch_size,
                device=self.device,
            )
            subgraph_edges = epoch_dataset.n_edges / epoch_dataset.size
            self.writer.add_scalar('subgraph_edges', subgraph_edges, epoch)

            # train epoch
            fit_loss, _ = self.fit(epoch_dataset.dataloader, optimizer)
            self.writer.add_scalar('fit_loss', fit_loss, epoch)

            # evaluate on epoch dataset
            train_loss, train_preds = self.score(epoch_dataset.dataloader)
            ap_score = self.ap_score_fn(train_preds, epoch_dataset.ds_labels)
            f1_score = self.f1_score_fn(train_preds, epoch_dataset.ds_labels)
            self.writer.add_scalar('train_loss', train_loss, epoch)
            self.writer.add_scalar('train_precision', ap_score, epoch)
            self.writer.add_scalar('train_f1', f1_score, epoch)

            # evaluate on valid dataset for early stopping
            valid_loss, valid_preds = self.score(self.valid_dataset.dataloader)
            valid_ap = self.ap_score_fn(valid_preds, self.valid_dataset.ds_labels)
            self.writer.add_scalar('valid_loss', valid_loss, epoch)
            self.writer.add_scalar('valid_precision', valid_ap, epoch)

            # update track record
            if track:
                track_record = pd.concat([
                    track_record,
                    pd.Series({
                        "epoch": epoch,
                        "runtime": time.perf_counter() - start_time,
                        # subgraph
                        "subgraph_edges": subgraph_edges,
                        # fit
                        "fit_loss": fit_loss,
                        # train
                        "train_loss": train_loss,
                        "train_precision": ap_score.item(),
                        "train_f1": f1_score.item(),
                        # valid
                        "valid_loss": valid_loss,
                        "valid_precision": valid_ap.item(),
                    }).to_frame().T
                ], ignore_index=True)

            # stop early if no improvement for last epochs
            early_stopper(valid_loss, self.net)
            if self.args.early_stopping and early_stopper.early_stop:
                break
            # stop with good precision
            if self.args.threshold_stopping and valid_ap > 0.99:
                break

        # load best model
        self.net.load_state_dict(torch.load('./out/model.pt'))
        return track_record

    def test(self, epoch: Union[int, None] = None) -> Tuple[float, float, float, float]:
        # evaluate on test set
        test_loss, test_preds = self.score(self.test_dataset.dataloader)
        test_ap = self.ap_score_fn(test_preds, self.test_dataset.ds_labels)
        test_f1 = self.f1_score_fn(test_preds, self.test_dataset.ds_labels)  # score at threshold=0.5
        test_conf = self.bcm_fn(test_preds, self.test_dataset.ds_labels)
        self.writer.add_scalar('test_precision', test_ap, epoch)
        self.writer.add_scalar('test_f1', test_f1, epoch)
        self.writer.add_text(
            "test_confusion",
            f"""
                non-edge | right: {test_conf[0][0]} / wrong: {test_conf[0][1]} / all: {self.test_dataset.n_non_edges}
                edge | right: {test_conf[1][1]} / wrong: {test_conf[1][0]} / all: {self.test_dataset.n_edges}
            """,
            epoch
        )

        # get best threshold
        precision, recall, thresholds = self.pr_curve_fn(test_preds, self.test_dataset.ds_labels)
        f1_scores = (2 * precision * recall) / (precision + recall)
        threshold = thresholds[np.argmax(f1_scores.cpu())].item()

        return test_loss, test_ap.item(), test_f1.item(), threshold

    def eval(self, toroid: bool = False) -> plt.Figure:
        # evaluate whole dataset (in batches)
        eval_loss, eval_preds = self.score(self.whole_dataset.dataloader)
        eval_ap = self.ap_score_fn(eval_preds, self.whole_dataset.ds_labels)

        # get best threshold
        precision, recall, thresholds = self.pr_curve_fn(eval_preds, self.whole_dataset.ds_labels)
        f1_scores = (2 * precision * recall) / (precision + recall)
        threshold = thresholds[np.argmax(f1_scores.cpu())].item()

        node_pairs = [
            (u, v)
            for ((u, _), (v, _)) in self.whole_dataset.node_feature_pairs
        ]

        # plot the graph prediction
        fig_size = 10
        fig, axs = plt.subplots(
            ncols=2,
            figsize=(fig_size, fig_size / 2.0)
        )

        axs[0].set_title(f"Original")
        visualization.draw_graph(
            ax=axs[0],
            graph=self.graph,
            toroid=toroid,
        )
        axs[1].set_title(f"Reconstruction [score={np.round(eval_ap.item(), decimals=4)}]")
        visualization.draw_prediction(
            ax=axs[1],
            graph=self.graph,
            prediction={node_pairs[i]: p for i, p in enumerate(eval_preds.cpu().numpy())},
            threshold=threshold,
            toroid=toroid,
        )
        visualization.draw_cbar(
            fig=fig,
            ax=axs[1],
            label=f"Prediction",
            threshold=threshold
        )
        return fig
