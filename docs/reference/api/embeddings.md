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

Gender bias as a direction in the embedding space: the gender axis recovered as
the top principal component of the definitional-pair differences, hard debiasing
via `neutralize` and `equalize`, WEAT's association score with its exact
enumerated permutation p-value, and the neutral-neighbour map that shows the
indirect bias surviving the projection.

::: dlhub.embeddings.debiasing_word_embeddings
    options:
      show_submodules: false
