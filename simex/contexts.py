from typing import List, Tuple, Dict
from sortedcontainers import SortedSet, SortedDict


def get_probe_contexts(probes: SortedSet,
                       tokens: List[str],
                       context_size: int,
                       preserve_order: bool,
                       min_num_contexts: int = 2,
                       exclude_punctuation: bool = False,
                       ) -> Tuple[Dict[str, Tuple[str]], SortedSet, SortedSet]:
    # get all probe contexts
    probe2contexts = SortedDict({p: [] for p in probes})
    context_types = SortedSet()
    for n, target in enumerate(tokens[:-context_size]):

        if target not in probes:
            continue

        left = tokens[n - context_size:n]
        right = tokens[n + 1:n + 1 + context_size]

        if exclude_punctuation:
            if '.' in left or '<eos>' in left:
                left = []
            if '.' in right or '<eos>' in right:
                right = []

        context = tuple(left + right)
        if not context:
            continue

        print(target, context)

        # collect
        if preserve_order:
            probe2contexts[target].append(context)
            context_types.add(context)
        else:
            single_word_contexts = [(w,) for w in context]
            probe2contexts[target].extend(single_word_contexts)
            context_types.update(single_word_contexts)

    # exclude entries with too few contexts
    excluded = []
    included = SortedSet()
    for probe, contexts in probe2contexts.items():
        if len(contexts) < min_num_contexts:
            excluded.append(probe)
        else:
            included.add(probe)
    for p in excluded:
        print(f'WARNING: Excluding "{p}" because it occurs {len(probe2contexts[p])} times')
        del probe2contexts[p]

    return probe2contexts, context_types, included