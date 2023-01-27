from __future__ import annotations

import random
import math
import numpy as np
import networkx as nx

from src.dict import list_to_dict


def gen_disc(amount: int) -> list[(float, float)]:
    points = []
    while len(points) < amount:
        p = (random.uniform(0, 1), random.uniform(0, 1))
        d = (p[0] - 0.5, p[1] - 0.5)
        if math.sqrt(d[0] * d[0] + d[1] * d[1]) > 0.5:
            continue
        points.append(p)
    return points

## use this to train nn (uniform information per distance)
## change resolution around certain distances (maybe around threshold)

def gen_disc_graph(size: int, avg_degree: int, radius: int = 0):
    # either use the provided radius or calculate based on avg_degree
    rg_radius = radius if radius > 0 else math.sqrt(avg_degree / (size * math.pi))
    pos = list_to_dict(gen_disc(size))
    return nx.random_geometric_graph(size, rg_radius, pos=pos), pos, rg_radius

def gen_disc_edge(d: float) -> [float, float, float, float]:
    while True:
        px, py = random.uniform(0, 1), random.uniform(0, 1)
        v = np.random.random(2)
        vd = (v / np.linalg.norm(v)) * d
        [qx, qy] = [px, py] + vd
        if math.dist([px, py], [0.5, 0.5]) <= 0.5 and math.dist([qx, qy], [0.5, 0.5]) <= 0.5:
            return [px, py, qx, qy]