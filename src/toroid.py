import math
from copy import deepcopy
import networkx as nx
from typing import Union

node_shifts = {
    "none": (0, 0),
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, -1),
    "right": (0, 1),
    "top_left": (1, -1),
    "top_right": (1, 1),
    "bottom_left": (-1, -1),
    "bottom_right": (-1, 1),
}


def apply_shift(key: str, shift: str) -> str:
    (sx, sy) = node_shifts[key]
    (ssx, ssy) = node_shifts[shift]
    # find shift that matches double shifted delta
    shifts = [k for i, [k, (vx, vy)] in enumerate(node_shifts.items()) if vx == sx + ssx and vy == sy + ssy]
    if len(shifts) == 0:
        return ""
    return shifts[0]


def on_toroid(graph: nx.Graph) -> nx.Graph:
    # create a copy of the graph to ensure that the original graph is not altered for the representation
    graph = deepcopy(graph)
    t_graph = nx.Graph()

    # add shifted copies for all nodes
    # the original node key is also replaced with the shift suffix 'none'
    t_graph.add_nodes_from([
        (
            f"{u}_{key}",
            {
                **d,
                "pos": [d["pos"][0] + xs, d["pos"][1] + ys],
                "shift": key,
                "original": u
            }
        )
        for (u, d) in graph.nodes(data=True)
        for (key, (xs, ys)) in node_shifts.items()
    ])

    # add one edge for each original edge
    # for each possible shift, use the edge that is the shortest
    positions = t_graph.nodes.data("pos")
    shortest_edges = [
        sorted([
            (u, v, {
                **d,
                "shift": shift,
                "dist": math.dist(
                    positions[f"{u}_none"],
                    (positions[f"{v}_none"][0] + sx, positions[f"{v}_none"][1] + sy)
                )
            })
            for [shift, (sx, sy)] in node_shifts.items()
        ], key=lambda x: x[2]["dist"])[0]
        for (u, v, d) in graph.edges(data=True)
    ]

    # for each shifted edge, add two halve edges u -> v' and u' -> v
    # un-shifted edges are added twice, but that should not matter
    t_graph.add_edges_from([
        (
            f"{u}_{shift}",
            f"{v}_{double_shift}",
            d
        )
        for (u, v, d) in shortest_edges
        for [shift, _] in node_shifts.items()
        if (double_shift := apply_shift(shift, d['shift'])) != ""
    ])

    # remove shifted nodes that are not of use
    t_graph.remove_nodes_from([
        u
        for u, p in t_graph.nodes(data="pos")
        if t_graph.degree(u) == 0 and (1.0 < p[0] or p[0] < 0.0 or 1.0 < p[1] or p[1] < 0.0)
    ])

    return t_graph


def toroid_edge(node: Union[str, int]) -> str:
    return node.split("_", 1)[1]


def is_shifted(node: Union[str, int]) -> bool:
    return str(node).count("_") == 1


def unshifted(node: Union[str, int]) -> int:
    return int(str(node).split("_", 1)[0])
