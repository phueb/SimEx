import resource
from typing import List, Optional, Tuple
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics.pairwise import cosine_similarity
from sortedcontainers import SortedSet

from simex.contexts import get_probe_contexts
from simex.representation import make_probe_vectors
from simex.utils import PipeLineResult, cluster


def do_pipeline(tokens: List[str],
                probes: SortedSet,
                context_size: int = 1,
                min_num_contexts: int = 1,
                preserve_word_order: bool = True,
                exclude_punctuation: bool = True,
                num_sing_dims: Optional[int] = None,
                ) -> Tuple[PipeLineResult, PipeLineResult]:
    """
    return two clustered similarity matrices, one prior to SVD, and another after SVD

    """

    maxsize = 1024 * 1024 * 1024 * 4
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))

    # get representations (with word-order)
    probe2contexts, context_types, probes = get_probe_contexts(probes,
                                                               tokens,
                                                               context_size,
                                                               preserve_word_order,
                                                               min_num_contexts,
                                                               exclude_punctuation,
                                                               )

    # for k, v in probe2contexts.items()[:10]:
    #     print(k)
    #     print(v)

    probe_reps1 = make_probe_vectors(probe2contexts, context_types)
    print('shape of probe representations={}'.format(probe_reps1.shape))

    # svd (careful with memory)
    if num_sing_dims is None:
        num_sing_dims = probe_reps1.shape[1] // 2
    probe_reps2, _, _ = randomized_svd(probe_reps1, num_sing_dims)
    print('shape after SVD={}'.format(probe_reps2.shape))

    # correlation
    sim_mat1 = cosine_similarity(probe_reps1)
    sim_mat2 = cosine_similarity(probe_reps2)

    # cluster
    result1 = cluster(sim_mat1, None, None, probes, probes)
    result2 = cluster(sim_mat2, None, None, probes, probes)

    print('Completed pipeline')

    return result1, result2
