from typing import List
from sortedcontainers import SortedSet
import streamlit as st
import altair as alt
import pandas as pd

from simex import configs
from simex.io import load_tokens
from simex.utils import PipeLineResult, to_columnar
from simex.pipeline import do_pipeline

CONTEXT_SIZE = 1
PRESERVE_WORD_ORDER = bool(1)
NUM_SING_DIMS = 12
MIN_NUM_CONTEXTS = 100
DEFAULT_NUM_DIM = 8


@st.cache
def load_corpus() -> List[str]:
    return load_tokens('childes-20201026')


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


# load data
tokens = load_corpus()

#########################################################
# GUI
#########################################################

# sidebar
st.sidebar.title('SimEx')
st.sidebar.write('Explore distributional similarities between words in AO-CHILDES.')

num_dims_select = list(range(1, 16))
num_dims = st.sidebar.selectbox('Select the number of singular dimensions to keep.',
                                num_dims_select, index=DEFAULT_NUM_DIM)

exclude_punctuation = st.sidebar. checkbox('Exclude punctuation from word contexts')

st.sidebar.write("""
         This visualization is part of a research effort into the distributional structure of nouns in child-directed speech. 
         More info can be found at http://languagelearninglab.org/
     """)


#########################################################
# computation
#########################################################

# do computation
pr1, pr2 = do_pipeline(tokens,
                       probes,
                       CONTEXT_SIZE,
                       MIN_NUM_CONTEXTS,
                       PRESERVE_WORD_ORDER,
                       exclude_punctuation,
                       num_dims,
                       )

pr1: PipeLineResult
pr2: PipeLineResult

# convert matrix to data frame
mat_df1 = pd.DataFrame(data=to_columnar(pr1.clustered_sim_mat))
mat_df2 = pd.DataFrame(data=to_columnar(pr2.clustered_sim_mat))


#########################################################
# show results
#########################################################


# color scale
scale = alt.Scale(
    domain=[-1, +1],
)


# before svd
heat_chart1 = alt.Chart(mat_df1).mark_rect().encode(
    alt.X('x:O', axis=None),
    alt.Y('y:O', axis=None),
    color=alt.Color('c:Q', scale=scale),
).properties(
    width=configs.Heatmap.width,
    height=configs.Heatmap.width,
)


# after svd
heat_chart2 = alt.Chart(mat_df2).mark_rect().encode(
    alt.X('x:O', axis=None),
    alt.Y('y:O', axis=None),
    color=alt.Color('c:Q', scale=scale),
).properties(
    width=configs.Heatmap.width,
    height=configs.Heatmap.width,
)


st.header('Similarity')
col1, col2 = st.columns(2)

# before svd
with col1:
    st.header('Before SVD')
    st.altair_chart(heat_chart1)


# after svd
with col2:
    st.header('After SVD')
    st.altair_chart(heat_chart2)