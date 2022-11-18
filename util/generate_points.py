import random
import math


# https://stackoverflow.com/a/36460020/10619052
def list_to_dict(items: list) -> dict:
    return {v: k for v, k in enumerate(items)}


def gen_disc(amount: int) -> list[(float, float)]:
    points = []
    while len(points) < amount:
        p = (random.uniform(0, 1), random.uniform(0, 1))
        d = (p[0] - 0.5, p[1] - 0.5)
        if math.sqrt(d[0] * d[0] + d[1] * d[1]) > 0.5:
            continue
        points.append(p)
    return points


def gen_square(amount: int) -> list[(float, float)]:
    return [(random.uniform(0, 1), random.uniform(0, 1)) for _ in range(amount)]
