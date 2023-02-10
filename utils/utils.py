import numpy as np
from typing import Dict, List, Optional, Tuple

def factorize_shapes(  # noqa C901
    n: int, d: int = 3, allow_round_up: bool = True
) -> List[int]:
    from itertools import cycle, islice

    # pyre-fixme[21]
    from scipy.stats import entropy
    from sympy.ntheory import factorint
    from sympy.utilities.iterables import multiset_partitions

    def _auto_shape(n: int, d: int = 3) -> List[int]:
        def _to_list(x: Dict[int, int]) -> List[int]:
            res = []
            for k, v in x.items():
                res += [k] * v
            return res

        p = _to_list(factorint(n))
        if len(p) < d:
            p = p + [1] * (d - len(p))

        def _roundrobin(*iterables):
            pending = len(iterables)
            nexts = cycle(iter(it).__next__ for it in iterables)
            while pending:
                try:
                    for next in nexts:
                        yield next()
                except StopIteration:
                    pending -= 1
                    nexts = cycle(islice(nexts, pending))

        def prepr(x: List[int]) -> Tuple:
            x = sorted(np.prod(_) for _ in x)
            N = len(x)
            xf, xl = x[: N // 2], x[N // 2 :]
            return tuple(_roundrobin(xf, xl))

        raw_factors = multiset_partitions(p, d)
        clean_factors = [prepr(f) for f in raw_factors]
        factors = list(set(clean_factors))
        # pyre-fixme[16]
        weights = [entropy(f) for f in factors]
        i = np.argmax(weights)
        return list(factors[i])

    def _roundup(n: int, k: int) -> int:
        return int(np.ceil(n / 10**k)) * 10**k

    if allow_round_up:
        weights = []
        for i in range(len(str(n))):
            n_i = _roundup(n, i)
            # pyre-fixme[16]
            weights.append(entropy(_auto_shape(n_i, d=d)))
        i = np.argmax(weights)
        factors = _auto_shape(_roundup(n, i), d=d)
    else:
        factors = _auto_shape(n, d=d)
    return factors