from copy import deepcopy
import networkx as nx
from typing import Tuple, Union, Dict

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap

from src.toroid import on_toroid, unshifted


# seaborn settings
sns.set_context("paper")
sns.set_style("darkgrid", {"grid.color": ".8"})
palette = "Spectral"
cmap = sns.color_palette(palette, as_cmap=True)


def draw_graph(
        ax: plt.Axes,
        graph: nx.Graph,
        subgraph: Union[nx.Graph, None] = None,
        toroid: bool = False
):
    if toroid:
        graph = on_toroid(graph)
    # limit graph (in case the graph is on a toroid)
    ax.set_aspect('equal')
    ax.set_xlim(left=0, right=1)
    ax.set_ylim(bottom=0, top=1)
    # set edge colors for subgraph nodes and edges
    node_color = [
        cmap(0.0) if subgraph.has_node(unshifted(u)) else cmap(1.0)
        for u in list(graph.nodes)
    ] if subgraph is not None else cmap(1.0)
    edge_color = [
        cmap(0.1) if subgraph.has_edge(unshifted(u), unshifted(v)) else cmap(0.9)
        for (u, v) in graph.edges
    ] if subgraph is not None else cmap(0.9)
    # draw graph
    nx.draw_networkx(
        graph,
        pos=graph.nodes.data("pos"),
        node_color=node_color,
        edge_color=edge_color,
        with_labels=False,
        node_size=1.5,
        ax=ax
    )


def draw_prediction(
        ax: plt.Axes,
        graph: nx.Graph,
        prediction: Dict[Tuple[int, int], float],
        threshold: float = 0.0,
        toroid: bool = False
):
    # remove all edges that have a prediction below the threshold
    graph = deepcopy(graph)
    graph.remove_edges_from([
        edge
        for edge in graph.edges
        if edge in prediction and prediction[edge] < threshold
    ])
    if toroid:
        graph = on_toroid(graph)
    # limit graph (in case the graph is on a toroid)
    ax.set_aspect('equal')
    ax.set_xlim(left=0, right=1)
    ax.set_ylim(bottom=0, top=1)
    # set edge colors for subgraph nodes and edges
    node_color = [
        cmap(1.0)
        for _ in graph.nodes
    ]
    edge_color = [
        cmap(prediction[(unshifted(u), unshifted(v))])
        for (u, v) in graph.edges
    ]
    # draw graph
    nx.draw_networkx(
        graph,
        pos=graph.nodes.data("pos"),
        node_color=node_color,
        edge_color=edge_color,
        with_labels=False,
        node_size=1.5,
        ax=ax
    )


def draw_cbar(
        fig: plt.Figure,
        ax: plt.Axes,
        label: str = "",
        threshold: float = 0.0
):
    # remove all color below threshold and rescale color bar
    th_colors = cmap(np.linspace(threshold, 1, 256))
    th_cmap = ListedColormap(th_colors)
    mapper = mpl.cm.ScalarMappable(cmap=th_cmap)
    mapper.set_clim(threshold, 1.0)

    cax = ax.inset_axes([1.04, 0.0, 0.05, 1.0])
    fig.colorbar(mapper, cax=cax, label=label)
