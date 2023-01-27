from __future__ import annotations

import random

def gen_square(amount: int) -> list[(float, float)]:
    return [(random.uniform(0, 1), random.uniform(0, 1)) for _ in range(amount)]
