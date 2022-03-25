from sortedcontainers import SortedSet

from simex.io import load_tokens
from simex.pipeline import do_pipeline
from simex.figs import plot_heatmap

CONTEXT_SIZE = 1
PRESERVE_WORD_ORDER = bool(1)
EXCLUDE_PUNCTUATION = bool(1)
NUM_SING_DIMS = 12
MIN_NUM_CONTEXTS = 100

tokens = load_tokens('childes-20201026')


probes = ['cat', 'dog',
          'mom', 'dad',
          'hand', 'foot',
          'tv', 'radio',
          'her', 'his',
          'here', 'there',
          'this', 'that',
          'green', 'red',
          'running', 'walking',
          'talk', 'say',
          ]

probes = SortedSet(probes)

# do computation
pr1, pr2 = do_pipeline(tokens,
                       probes,
                       CONTEXT_SIZE,
                       MIN_NUM_CONTEXTS,
                       PRESERVE_WORD_ORDER,
                       EXCLUDE_PUNCTUATION,
                       NUM_SING_DIMS,
                       )

# plot
plot_heatmap(pr1.clustered_sim_mat,
             pr1.row_labels,
             pr1.col_labels,
             label_interval=1,
             dpi=192 * 4,
             ax_font_size=8)

