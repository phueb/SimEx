
from collections import Counter
from typing import Set, Optional, Tuple, Dict

import numpy as np
import pyprind
from sortedcontainers import SortedSet


def make_probe_reps_median_split(probe2contexts: Dict[str, Tuple[str]],
                                 context_types: SortedSet,
                                 split_id: int,
                                 ) -> np.ndarray:
    """
    make probe representations based on first or second median split of each probe's contexts.
    representation can be BOW or preserve word-order, depending on how contexts were collected.
    """
    num_context_types = len(context_types)
    probes = SortedSet(probe2contexts.keys())
    assert '' not in probes

    num_probes = len(probe2contexts)
    context2col_id = {c: n for n, c in enumerate(context_types)}

    probe_reps = np.zeros((num_probes, num_context_types))
    progress_bar = pyprind.ProgBar(num_probes, stream=2, title='Making representations form contexts')
    for row_id, p in enumerate(probes):
        probe_contexts = probe2contexts[p]
        num_probe_contexts = len(probe_contexts)
        num_in_split = num_probe_contexts // 2

        if len(probe_contexts) < 2:  # otherwise, cannot split
            raise RuntimeError(f'WARNING: "{p}" has less than 2 contexts ({probe_contexts})')

        # get either first half or second half of contexts
        if split_id == 0:
            probe_contexts_split = probe_contexts[:num_in_split]
        elif split_id == 1:
            probe_contexts_split = probe_contexts[-num_in_split:]
        else:
            raise AttributeError('Invalid arg to split_id.')

        # make probe representation
        c2f = Counter(probe_contexts_split)
        for c, f in c2f.items():
            col_id = context2col_id[c]
            probe_reps[row_id, col_id] = f

        progress_bar.update()

    # check each representation has information
    num_zero_rows = np.sum(~probe_reps.any(axis=1))
    assert num_zero_rows == 0

    return probe_reps


