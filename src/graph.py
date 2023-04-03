import itertools
import math
import numpy as np
import random
import networkx as nx
from typing import Set, Tuple, Union, Dict
from girg_sampling import girgs
from copy import deepcopy

from src.args import Args

# Type alias to make the code more comprehensible
NodeData = Dict
NodeId = int
NodeWithData = Tuple[NodeId, NodeData]
NodeDataPairs = Set[Tuple[NodeWithData, NodeWithData]]


node_shifts = {
    "top": (1, 0),
    "down": (-1, 0),
    "left": (0, -1),
    "right": (0, 1),
    "top-left": (1, -1),
    "top-right": (1, 1),
    "bottom-left": (-1, -1),
    "bottom-right": (-1, 1),
}


def gen_graph(args: Args) -> Tuple[nx.Graph, int]:
    if args.graph_type == "rgg":
        radius = math.sqrt(args.rg_avg_degree / (args.graph_size * math.pi))
        graph, dim = random_geometric_graph(
            size=args.graph_size,
            radius=radius
        )
    elif args.graph_type == "t_rgg":
        radius = math.sqrt(args.rg_avg_degree / (args.graph_size * math.pi))
        graph, dim = toroid_random_geometric_graph(
            size=args.graph_size,
            radius=radius
        )
    elif args.graph_type == "random":
        radius = math.sqrt(args.rg_avg_degree / (args.graph_size * math.pi))
        graph, dim = random_geometric_graph(
            size=args.graph_size,
            radius=radius
        )
        randomize_features(graph)
    elif args.graph_type == "girg":
        graph, dim = girg_graph(
            size=args.graph_size
        )
    else:
        raise ValueError(f"invalid graph type: {args.graph_type}")
    return graph, dim


def girg_graph(
        size: int,
        ple: float = 2.5,
        alpha: float = math.inf,
        deg: float = 10,
) -> Tuple[nx.Graph, int]:
    # generate girg
    weights = girgs.generateWeights(size, ple)
    positions = girgs.generatePositions(size, 2)
    weights_scale = girgs.scaleWeights(weights, deg, 2, alpha)
    scaled_weights = [x * weights_scale for x in weights]
    edges = girgs.generateEdges(scaled_weights, positions, alpha)

    # convert to NetworkX graph (use pos and weight as feature)
    node_positions = {i: pos for i, pos in enumerate(positions)}
    node_features = {
        i: (*positions[i], scaled_weights[i])
        for i in range(size)
    }
    graph = nx.Graph()
    graph.add_nodes_from(range(size))
    nx.set_node_attributes(graph, node_positions, name="pos")
    nx.set_node_attributes(graph, node_features, name="feature")
    graph.add_edges_from(edges)
    return graph, 3


def gen_disc_pos(disc: bool = True) -> Tuple[float, float]:
    while True:
        p = (random.uniform(0, 1), random.uniform(0, 1))
        d = (p[0] - 0.5, p[1] - 0.5)
        if disc and math.sqrt(d[0] * d[0] + d[1] * d[1]) > 0.5:
            continue
        return p


def random_geometric_graph(size: int, radius: float, disc: bool = True) -> Tuple[nx.Graph, int]:
    # generate graph from positions and radius
    node_positions = {i: gen_disc_pos(disc=disc) for i in range(size)}
    graph = nx.random_geometric_graph(
        size,
        radius,
        pos=node_positions
    )
    nx.set_node_attributes(graph, node_positions, name="pos")
    nx.set_node_attributes(graph, node_positions, name="feature")
    return graph, 2


def toroid_random_geometric_graph(size: int, radius: float) -> Tuple[nx.Graph, int]:
    graph, dim = random_geometric_graph(size, radius, disc=False)
    # add additional edges
    graph.add_edges_from([
        (u, v, {"shift": key})
        for ((u, p_u), (v, (p_v_x, p_v_y))) in itertools.combinations(graph.nodes(data="pos"), 2)
        for (key, (s_x, s_y)) in node_shifts.items()
        if np.abs(math.dist(p_u, (p_v_x + s_x, p_v_y + s_y))) <= radius
    ])
    return graph, 2


def randomize_features(graph: nx.Graph):
    node_positions = {i: gen_disc_pos() for i in list(graph.nodes)}
    nx.set_node_attributes(graph, node_positions, name="feature")


def subgraph(
        graph: nx.Graph,
        size: int,
        alpha: float = 1.0,
        boredom_pth: float = 0.3,
        ignore=None
) -> Tuple[nx.Graph, Set[NodeId]]:
    # init
    curr_node: Union[NodeId, None] = None
    sampled: Set[NodeId] = set()
    boredom: int = 0  # escape parameter in case it walks on a too small component, enforces jumps
    while len(sampled) < size:
        neighbors = list(graph.neighbors(curr_node)) if curr_node is not None else []
        # random jump (first iteration is always a jump)
        if len(neighbors) == 0 \
                or random.random() < alpha \
                or (boredom > size * boredom_pth):
            next_node = random.choice(list(graph.nodes))
        # random walk
        else:
            next_node = random.choice(neighbors)
        # skip ignore list (don't even allow walking over it)
        if ignore is not None and next_node in ignore:
            boredom = boredom + 1
            continue
        curr_node = next_node
        # add found node if new, otherwise increase boredom
        if curr_node in sampled:
            boredom = boredom + 1
            continue
        boredom = 0
        sampled.add(curr_node)
    # build subgraph
    return graph.subgraph(sampled), sampled


def periodic_of(graph: nx.Graph) -> nx.Graph:
    graph = deepcopy(graph)
    positions = graph.nodes.data("pos")
    # add shifted copies for all nodes
    graph.add_nodes_from([
        (
            f"{n}_{key}",
            {**d, "pos": [d["pos"][0] + xs, d["pos"][1] + ys]}
        )
        for (n, d) in graph.nodes(data=True)
        for (key, (xs, ys)) in node_shifts.items()
    ])
    # add edges between all original nodes and their shifts
    # this has to be done for both directions
    graph.add_edges_from([
        (
            u if lr else v,
            f"{v if lr else u}_{key}",
            {**d, "shift": key}
        )
        for (u, v, d) in graph.edges(data=True)
        for (key, shift) in node_shifts.items()
        for lr in [True, False]
    ])
    # remove all edges that are too long and unused nodes
    graph.remove_edges_from([
        (u, v)
        for (u, v, d) in graph.edges(data=True)
        if np.abs(math.dist(positions[u], positions[v])) > 0.5
    ])
    graph.remove_nodes_from([
        u
        for (u, d) in graph.nodes(data=True)
        if (d["pos"][0] > 1.0 or d["pos"][1] > 1.0 or d["pos"][0] < 0.0 or d["pos"][1] < 0.0) and graph.degree(u) == 0
    ])
    return graph


def non_periodic_node(node: Union[int, str]) -> int:
    return int(str(node).split("_", 1)[0])
