import itertools
import math
from copy import deepcopy

import numpy as np
import random
import networkx as nx
from typing import Tuple, Union
from girg_sampling import girgs

from src.args import Args
from src.toroid import node_shifts


# Graph generation


def gen_pos() -> Tuple[float, float]:
    return random.uniform(0, 1), random.uniform(0, 1)


def gen_edge(d: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    while True:
        u, v = gen_pos(), gen_pos()
        uv = tuple(np.subtract(v, u))
        # 'w' is moved along line 'uv'
        w = np.add(u, (uv / np.linalg.norm(uv)) * d)
        # continue until 'w' is inside the area
        if 0 > w[0] > 1 or 0 > w[1] > 1:
            continue
        return u, w


# TODO do not use 'args' struct here? Keep in line with subgraph generation methods
def gen_graph(args: Args) -> nx.Graph:
    if args.graph_type == "rgg":
        graph = gen_rgg(
            size=args.graph_size,
            avg_degree=args.rgg_avg_degree
        )
    elif args.graph_type == "t_rgg":
        graph = gen_t_rgg(
            size=args.graph_size,
            avg_degree=args.rgg_avg_degree
        )
    elif args.graph_type == "random":
        graph = gen_rgg(
            size=args.graph_size,
            avg_degree=args.rgg_avg_degree
        )
        graph = randomized_graph(graph)
    elif args.graph_type == "girg":
        graph = gen_girg(
            size=args.graph_size
        )
    else:
        raise ValueError(f"invalid graph type: {args.graph_type}")
    return graph


def gen_girg(
        size: int,
        ple: float = 2.5,
        alpha: float = math.inf,
        deg: float = 10,
) -> nx.Graph:
    # generate girg
    weights = girgs.generateWeights(size, ple)
    positions = girgs.generatePositions(size, 2)
    weights_scale = girgs.scaleWeights(weights, deg, 2, alpha)
    scaled_weights = [x * weights_scale for x in weights]
    edges = girgs.generateEdges(scaled_weights, positions, alpha)

    # convert to NetworkX graph (use pos and weight as feature)
    node_positions = {i: pos for i, pos in enumerate(positions)}
    weights_mean = np.mean(scaled_weights)
    norm_weights = [w / weights_mean for w in scaled_weights]  # "normalize" weights so that np.std does not overflow
    node_features = {
        i: (*positions[i], norm_weights[i])
        for i in range(size)
    }
    graph = nx.Graph()
    graph.add_nodes_from(range(size))
    nx.set_node_attributes(graph, node_positions, name="pos")
    nx.set_node_attributes(graph, node_features, name="feature")
    graph.add_edges_from(edges)
    return graph


def gen_rgg(size: int, avg_degree: Union[float, None], radius: Union[float, None] = None) -> nx.Graph:
    if radius is None:
        radius = math.sqrt(avg_degree / ((size - 1) * math.pi))
    # generate graph from positions and radius
    node_positions = {i: gen_pos() for i in range(size)}
    graph = nx.random_geometric_graph(
        size,
        radius,
        pos=node_positions
    )
    nx.set_node_attributes(graph, node_positions, name="pos")
    nx.set_node_attributes(graph, node_positions, name="feature")
    return graph


def gen_t_rgg(size: int, avg_degree: Union[float, None], radius: Union[float, None] = None) -> nx.Graph:
    if radius is None:
        radius = math.sqrt(avg_degree / ((size - 1) * math.pi))
    graph = gen_rgg(size, radius)
    # add additional edges
    graph.add_edges_from([
        (u, v, {"shift": key})
        for ((u, p_u), (v, (p_v_x, p_v_y))) in itertools.combinations(graph.nodes(data="pos"), 2)
        for (key, (s_x, s_y)) in node_shifts.items()
        if np.abs(math.dist(p_u, (p_v_x + s_x, p_v_y + s_y))) <= radius
    ])
    return graph


# Graph manipulation


def randomized_graph(graph: nx.Graph) -> nx.Graph:
    graph = deepcopy(graph)
    # overwrite node features that do not reflect the graph structure
    node_positions = {i: gen_pos() for i in list(graph.nodes)}
    nx.set_node_attributes(graph, node_positions, name="feature")
    return graph


def sorted_graph(graph: nx.Graph) -> nx.Graph:
    out = nx.Graph()
    out.add_nodes_from(sorted(
        list(graph.nodes(data=True)),
        key=lambda x: x[1]["feature"]
    ))
    out.add_edges_from(graph.edges(data=True))
    return out
