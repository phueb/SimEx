
from collections import Counter
from typing import Tuple, Dict

import numpy as np
import pyprind
from sortedcontainers import SortedSet


def make_probe_vectors(probe2contexts: Dict[str, Tuple[str]],
                       context_types: SortedSet,
                       ) -> np.ndarray:
    """
    make probe representations based on each probe's contexts.
    representation can be BOW or preserve word-order, depending on how contexts were collected.
    """
    num_context_types = len(context_types)
    probes = SortedSet(probe2contexts.keys())
    assert '' not in probes

    num_probes = len(probe2contexts)
    context2col_id = {c: n for n, c in enumerate(context_types)}

    res = np.zeros((num_probes, num_context_types), int)
    progress_bar = pyprind.ProgBar(num_probes, stream=2, title='Making representations form contexts')
    for row_id, p in enumerate(probes):
        probe_contexts = probe2contexts[p]

        # make probe representation
        c2f = Counter(probe_contexts)
        for c, f in c2f.items():
            col_id = context2col_id[c]
            res[row_id, col_id] = int(f)

        progress_bar.update()

    # check each representation has information
    num_zero_rows = np.sum(~res.any(axis=1))
    assert num_zero_rows == 0

    return res


