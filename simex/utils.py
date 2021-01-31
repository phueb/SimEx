import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from typing import Optional, Set
from cytoolz import itertoolz
from sortedcontainers import SortedSet


def split(l, split_size):
    for i in range(0, len(l), split_size):
        yield l[i:i + split_size]


def get_sliding_windows(window_size, tokens):
    res = list(itertoolz.sliding_window(window_size, tokens))
    return res


def to_corr_mat(data_mat):
    mns = data_mat.mean(axis=1, keepdims=True)
    stds = data_mat.std(axis=1, ddof=1, keepdims=True) + 1e-6  # prevent np.inf (happens when dividing by zero)
    deviated = data_mat - mns
    zscored = deviated / stds
    res = np.matmul(zscored, zscored.T) / len(data_mat)  # it matters which matrix is transposed
    return res


def cluster(mat: np.ndarray,
            dg0: Optional[dict],
            dg1: Optional[dict],
            original_row_words: Optional[SortedSet] = None,
            original_col_words: Optional[SortedSet] = None,
            method: str = 'complete',
            metric: str = 'cityblock'):

    print('Clustering...')
    if original_row_words is not None:
        assert len(original_row_words) == mat.shape[0]
    if original_col_words is not None:
        assert len(original_col_words) == mat.shape[1]

    if dg0 is None:
        lnk0 = linkage(mat, method=method, metric=metric, optimal_ordering=True)
        dg0 = dendrogram(lnk0,
                         ax=None,
                         color_threshold=None,
                         no_labels=True,
                         no_plot=True)
    res = mat[dg0['leaves'], :]  # reorder rows

    if dg1 is None:
        lnk1 = linkage(mat.T, method=method, metric=metric, optimal_ordering=True)
        dg1 = dendrogram(lnk1,
                         ax=None,
                         color_threshold=None,
                         no_labels=True,
                         no_plot=True)

    res = res[:, dg1['leaves']]  # reorder cols
    if original_row_words is None and original_col_words is None:
        return res, dg0, dg1
    else:
        row_labels = np.array(original_row_words)[dg0['leaves']]
        col_labels = np.array(original_col_words)[dg1['leaves']]
        return res, row_labels, col_labels, dg0, dg1