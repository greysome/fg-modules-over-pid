from functools import reduce
from typing import Iterable, Callable, Optional, Sequence

def alleq(L: Iterable, f: Optional[Callable] = None) -> bool:
    if f is None: return len(set(L)) <= 1
    else: return len(set(f(i) for i in L)) <= 1

def flatten(L: Sequence[Sequence]) -> Sequence:
    return [x for row in L for x in row]

def prod(L: Sequence, start):
    return reduce(lambda x,y: x*y, L, start)

def argmax(L: Sequence):
    max_idx = 0
    max_val = L[0]
    for idx, i in enumerate(L):
        if i > max_val:
            max_idx = idx
            max_val = i
    return max_idx