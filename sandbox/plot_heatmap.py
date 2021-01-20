"""
Research questions:
1. How well do nouns cluster together in hierarchical cluster tree generated from corpus statistics?
"""
from sklearn.decomposition import PCA

from categoryeval.probestore import ProbeStore

from simex.docs import load_docs
from simex.representation import make_probe_reps_median_split
from simex.contexts import get_probe_contexts
from simex.figs import plot_heatmap
from simex.utils import to_corr_mat, cluster


# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20191206'
PROBES_NAME = 'sem-4096'  # careful: some probe reps might be zero vectors if they do not occur in part


docs = load_docs(CORPUS_NAME)
probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id)


# ///////////////////////////////////////////////////////////////// parameters

CONTEXT_SIZE = 1
METRIC = 'ck'
PRESERVE_WORD_ORDER = False
N_COMPONENTS = 32

# get representations (with word-order)
probe2contexts, context_types = get_probe_contexts(probe_store.types,  # TODO this returns LEFT contexts only - make options for right context
                                                   prep.store.tokens,
                                                   CONTEXT_SIZE,
                                                   PRESERVE_WORD_ORDER)

dg0, dg1 = None, None
for part_id in range(2):
    probe_reps = make_probe_reps_median_split(probe2contexts, context_types, part_id)

    print('shape of reps={}'.format(probe_reps.shape))

    # pca
    pca = PCA(n_components=N_COMPONENTS)
    probe_reps = pca.fit_transform(probe_reps)
    print('shape after PCA={}'.format(probe_reps.shape))

    # plot
    corr_mat = to_corr_mat(probe_reps)
    print('shape of corr_mat={}'.format(corr_mat.shape))
    clustered_corr_mat, rls, cls, dg0, dg1 = cluster(corr_mat, dg0, dg1, prep.store.types, prep.store.types)
    rls = [rl if rl in probe_store.types else '' for rl in rls]
    plot_heatmap(clustered_corr_mat, rls, cls, label_interval=1)

