"""
Research questions:
1. How well do nouns cluster together in hierarchical cluster tree generated from corpus statistics?
"""
from sklearn.decomposition import PCA

from simex.io import load_tokens, load_probes
from simex.representation import make_probe_reps_median_split
from simex.contexts import get_probe_contexts
from simex.figs import plot_heatmap
from simex.utils import to_corr_mat, cluster


CORPUS_NAME = 'childes-20201026'
PROBES_NAME = 'nouns-2972'
CONTEXT_SIZE = 4
PRESERVE_WORD_ORDER = False
N_COMPONENTS = 32
PART_IDS = [0, 1]  # order matters
MIN_NUM_CONTEXTS = 1000

tokens = load_tokens(CORPUS_NAME)

probes = load_probes(PROBES_NAME)

# get representations (with word-order)
probe2contexts, context_types, probes = get_probe_contexts(probes,  # TODO this returns LEFT contexts only - make options for right context
                                                           tokens,
                                                           CONTEXT_SIZE,
                                                           PRESERVE_WORD_ORDER,
                                                           MIN_NUM_CONTEXTS)

dg0, dg1 = None, None
for part_id in PART_IDS:

    # todo median split doesn't guarantee that eah split falls neatly in part 1 vs. part2.
    # todo it is possible that contexts only occur at very end or beginning,
    #  so that median split doesn't represent split between partitions
    probe_reps = make_probe_reps_median_split(probe2contexts, context_types, part_id)

    print('shape of reps={}'.format(probe_reps.shape))

    # pca
    pca = PCA(n_components=N_COMPONENTS)
    probe_reps = pca.fit_transform(probe_reps)
    print('shape after PCA={}'.format(probe_reps.shape))

    # correlation
    corr_mat = to_corr_mat(probe_reps)
    print('shape of corr_mat={}'.format(corr_mat.shape))

    # cluster
    clustered_corr_mat, rls, cls, dg0, dg1 = cluster(corr_mat, dg0, dg1, probes, probes)

    # plot
    plot_heatmap(clustered_corr_mat, rls, cls, label_interval=1, dpi=192 * 4)

