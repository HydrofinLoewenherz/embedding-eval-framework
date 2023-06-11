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
cmap = sns.color_palette("Spectral", as_cmap=True)
pred_cmap = sns.color_palette("flare", as_cmap=True)
dark_cmap = sns.color_palette("Dark2", as_cmap=True)


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
    remap: bool = True,
    toroid: bool = False
):
    # remove all edges that have a prediction below the threshold
    graph = deepcopy(graph)

    vis_graph = nx.Graph()
    vis_graph.add_nodes_from(graph.nodes(data=True))
    vis_graph.add_edges_from(
        (u, v)
        for (u, v), p in prediction.items()
        if p >= threshold
    )

    if toroid:
        vis_graph = on_toroid(vis_graph)

    # limit graph (in case the graph is on a toroid)
    ax.set_aspect('equal')
    ax.set_xlim(left=0, right=1)
    ax.set_ylim(bottom=0, top=1)
    # set edge colors for subgraph nodes and edges
    node_color = [cmap(1.0)] * len(vis_graph.nodes)
    edge_color = [
        pred_cmap(
            # remap prediction into reduced colormap (map 'threshold' as 0)
            (prediction[(unshifted(u), unshifted(v))] - threshold) / (1 - threshold)
            if remap else
            prediction[(unshifted(u), unshifted(v))]
        )
        for (u, v) in vis_graph.edges
    ]
    # draw graph
    nx.draw_networkx(
        vis_graph,
        pos=vis_graph.nodes.data("pos"),
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
    threshold: float = 0.0,
    remap: bool = True
):
    # remove all color below threshold and rescale color bar
    if remap:
        mapper = mpl.cm.ScalarMappable(cmap=pred_cmap)
    else:
        th_colors = pred_cmap(np.linspace(threshold, 1, 256))
        th_cmap = ListedColormap(th_colors)
        mapper = mpl.cm.ScalarMappable(cmap=th_cmap)
    mapper.set_clim(threshold, 1.0)

    cax = ax.inset_axes([1.01, 0.0, 0.05, 1.0])
    fig.colorbar(mapper, cax=cax, label=label)


def draw_classification(
    ax: plt.Axes,
    graph: nx.Graph,
    prediction: Dict[Tuple[int, int], float],
    labels: Dict[Tuple[int, int], int],
    threshold: float = 0.0,
    toroid: bool = False
):
    graph = deepcopy(graph)

    vis_graph = nx.Graph()
    vis_graph.add_nodes_from(graph.nodes(data=True))
    vis_graph.add_edges_from(
        (u, v)
        for (u, v), p in prediction.items()
    )

    if toroid:
        vis_graph = on_toroid(vis_graph)

    # limit graph (in case the graph is on a toroid)
    ax.set_aspect('equal')
    ax.set_xlim(left=0, right=1)
    ax.set_ylim(bottom=0, top=1)

    colors = {
        # (true, pred): color
        (0, 0): (0.0, 0.0, 0.0, 0.0),  # transparent
        (1, 0): (0.0, 1.0, 0.0, 1.0),  # green
        (0, 1): (1.0, 0.0, 0.0, 1.0),  # red
        (1, 1): (0.0, 0.0, 1.0, 0.5),  # blue-transparent
    }
    edge_color = [
        colors[(
            labels[(u, v)],
            1 if p >= threshold else 0
        )]
        for (u, v), p in prediction.items()
    ]

    node_color = [cmap(0.1)] * len(vis_graph.nodes(data=False))
    # draw graph
    nx.draw_networkx(
        vis_graph,
        pos=vis_graph.nodes.data("pos"),
        node_color=node_color,
        edge_color=edge_color,
        with_labels=False,
        node_size=1.5,
        ax=ax
    )
    return edge_color
