import numpy as np
from typing import List, Union, Dict
from scipy.cluster.hierarchy import linkage, dendrogram
from typing import Optional
from dataclasses import dataclass
from sortedcontainers import SortedSet


@dataclass(frozen=True)
class PipeLineResult:
    clustered_sim_mat: np.array
    row_labels: Optional[List[str]]
    col_labels: Optional[List[str]]
    dg0: Dict
    dg1: Dict


def to_columnar(mat: np.array,
                row_labels: List[str],
                col_labels: List[str],
                ) -> Dict[str, List[Union[float, int]]]:
    """
    convert a matrix into a dict,
    where each entry corresponds to a single matrix element:
    - the row index
    - the col index
    - the element's value
      """
    res = {'context': [],
           'word': [],
           'sim': [],
           }

    for row_id, rw in enumerate(row_labels):
        for col_id, cw in enumerate(col_labels):
            res['word'].append(rw)
            res['context'].append(cw)
            res['sim'].append(mat[row_id, col_id])

    return res


def cluster(mat: np.ndarray,
            dg0: Optional[dict],
            dg1: Optional[dict],
            original_row_words: Optional[SortedSet] = None,
            original_col_words: Optional[SortedSet] = None,
            method: str = 'complete',
            metric: str = 'cosine') -> PipeLineResult:

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
        return PipeLineResult(clustered_sim_mat=res,
                              row_labels=None,
                              col_labels=None,
                              dg0=dg0,
                              dg1=dg1,
                              )
    else:
        row_labels = np.array(original_row_words)[dg0['leaves']]
        col_labels = np.array(original_col_words)[dg1['leaves']]
        return PipeLineResult(clustered_sim_mat=res,
                              row_labels=row_labels,
                              col_labels=col_labels,
                              dg0=dg0,
                              dg1=dg1,
                              )