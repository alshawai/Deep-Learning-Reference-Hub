# Word embeddings

<!-- hubcheck: generated -->

Static word vectors: the embedding matrix and lookup identity, the geometry of
word vectors (cosine similarity, nearest neighbors, the analogy property), and a
from-scratch PCA projection for visualization.

::: dlhub.embeddings.word_embeddings
    options:
      show_submodules: false

Learned word vectors: skip-gram with negative sampling (SGNS), CBOW, the
smoothed-unigram negative sampler, and GloVe's weighted least-squares objective,
each loss paired with its exact gradient, trained from scratch on a toy corpus.

::: dlhub.embeddings.word2vec_and_glove
    options:
      show_submodules: false
