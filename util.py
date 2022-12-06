from typing import Iterable, Callable, Optional, Sequence

def alleq(L: Iterable, f: Optional[Callable] = None) -> bool:
    if f is None: return len(set(L)) <= 1
    else: return len(set(f(i) for i in L)) <= 1

def flatten(L: Sequence[Sequence]) -> Sequence:
    return [x for row in L for x in row]