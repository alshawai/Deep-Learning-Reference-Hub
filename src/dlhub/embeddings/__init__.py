"""
Word Embeddings
===============

Static word vectors: the embedding matrix and lookup, the geometry of word
vectors (cosine similarity, nearest neighbors, the analogy property), and a
from-scratch PCA projection for visualization.

This concern sits beside ``optimizers``, ``tuning``, ``nn``, and ``training``
because word vectors are not neural layers -- the lookup is a data transformation
and the properties are linear algebra. How the vectors are *learned* is the
sibling ``word2vec-and-glove`` topic.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

from dlhub.embeddings.word_embeddings import (
    PCAProjection,
    analogy,
    cosine_similarity,
    lookup,
    make_fixture,
    nearest_neighbors,
    one_hot,
    pca_project_2d,
)

__all__ = [
    "PCAProjection",
    "analogy",
    "cosine_similarity",
    "lookup",
    "make_fixture",
    "nearest_neighbors",
    "one_hot",
    "pca_project_2d",
]
