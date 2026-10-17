"""
Word Embeddings
===============

Static word vectors in three parts. The first is what a word vector *is*: the
embedding matrix and lookup, the geometry of word vectors (cosine similarity,
nearest neighbors, the analogy property), and a from-scratch PCA projection for
visualization (:mod:`dlhub.embeddings.word_embeddings`). The second is how those
vectors are *learned*: skip-gram with negative sampling, CBOW, and GloVe, each
with its objective and exact gradient, trained on a toy corpus
(:mod:`dlhub.embeddings.word2vec_and_glove`). The third is what **bias** looks
like geometrically in a given embedding: the gender direction identified from
definitional pairs, Bolukbasi hard debiasing (neutralize and equalize), and the
WEAT measurement before and after
(:mod:`dlhub.embeddings.debiasing_word_embeddings`).

This concern sits beside ``optimizers``, ``tuning``, ``nn``, and ``training``
because word vectors are not neural layers -- the lookup is a data transformation
and the properties are linear algebra, while the learning objectives are plain
gradient descent on a scalar loss.

All three modules ship a ``make_fixture``. The head's bare ``make_fixture`` is
the one re-exported here; the learned corpus is reached as
``dlhub.embeddings.word2vec_and_glove.make_fixture`` and the planted bias matrix
as ``dlhub.embeddings.debiasing_word_embeddings.make_fixture``, so the three
fixtures do not collide at the package top level.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

from dlhub.embeddings.debiasing_word_embeddings import (
    equalize,
    gender_direction,
    hard_debias,
    neutral_neighbors,
    neutralize,
    weat_association,
    weat_effect_size,
    weat_pvalue,
    weat_statistic,
)
from dlhub.embeddings.word2vec_and_glove import (
    build_vocabulary,
    cbow_gradients,
    cbow_loss,
    cooccurrence_matrix,
    glove_gradients,
    glove_loss,
    glove_weight,
    nearest_neighbor,
    sample_negatives,
    sgns_gradients,
    sgns_loss,
    sigmoid,
    train_glove,
    train_sgns,
    training_pairs,
    unigram_probabilities,
)
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
    # What a word vector is (word_embeddings).
    "PCAProjection",
    "analogy",
    "cosine_similarity",
    "lookup",
    "make_fixture",
    "nearest_neighbors",
    "one_hot",
    "pca_project_2d",
    # How word vectors are learned (word2vec_and_glove).
    "build_vocabulary",
    "cbow_gradients",
    "cbow_loss",
    "cooccurrence_matrix",
    "glove_gradients",
    "glove_loss",
    "glove_weight",
    "nearest_neighbor",
    "sample_negatives",
    "sgns_gradients",
    "sgns_loss",
    "sigmoid",
    "train_glove",
    "train_sgns",
    "training_pairs",
    "unigram_probabilities",
    # The bias geometry of a given embedding (debiasing_word_embeddings).
    "equalize",
    "gender_direction",
    "hard_debias",
    "neutral_neighbors",
    "neutralize",
    "weat_association",
    "weat_effect_size",
    "weat_pvalue",
    "weat_statistic",
]
