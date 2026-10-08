"""
Word Embeddings -- Representation, Properties, and Projection
=============================================================

A from-scratch, teaching reference for *static* word embeddings: the embedding
matrix and the lookup identity, the geometry of word vectors (cosine similarity,
nearest neighbors, and the analogy / parallelogram property), and a from-scratch
PCA projection to two dimensions for visualization.

This module treats the vectors as **given** -- a small, hand-planted toy matrix
(see :func:`make_fixture`) whose every answer is exact and checkable by eye. How
embeddings are *learned* (skip-gram, CBOW, negative sampling, GloVe, fastText) is
a separate topic, and so are bias geometry and contextual embeddings. This is the
foundational head: what a word vector *is* and what you can read off it.

Notation follows Andrew Ng's Deep Learning Specialization, Course 5, Week 2. The
embedding matrix ``E`` has shape ``(d, V)`` -- ``d`` features (rows) by ``V``
vocabulary (columns). For a word with index ``i``, its one-hot column
``o_w in {0, 1}^V`` selects a column::

    e_w = E o_w = E[:, i]    in R^d

which is the module's spine: an embedding layer *is* a bias-free linear layer
whose input is one-hot, so in practice one gathers the column by index rather
than materializing the matrix product. (PyTorch stores the transpose, ``weight``
of shape ``(V, d)``, so there a word vector is a *row*, ``weight[i, :]``; the
idiom port documents that convention.)

References
----------
- Mikolov, T., Yih, W., & Zweig, G. (2013). Linguistic Regularities in
  Continuous Space Word Representations. NAACL-HLT, 746-751. (The analogy /
  parallelogram property.)
- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of
  Word Representations in Vector Space. https://arxiv.org/abs/1301.3781
- Pennington, J., Socher, R., & Manning, C. (2014). GloVe: Global Vectors for
  Word Representation. EMNLP. https://nlp.stanford.edu/pubs/glove.pdf
- Levy, O., Goldberg, Y., & Dagan, I. (2015). Improving Distributional Similarity
  with Lessons Learned from Word Embeddings. TACL, 3, 211-225. (3CosMul.)
- Ng, A. (2018). Deep Learning Specialization, Course 5 (Sequence Models),
  Week 2. https://www.coursera.org/specializations/deep-learning

Author
------
Deep Learning Reference Hub

License
-------
MIT

Notes
-----
- **Deterministic PCA sign.** The SVD fixes each principal direction only up to
  sign, so two runs could return mirror-image coordinates. :func:`pca_project_2d`
  removes the ambiguity with a fixed convention -- flip each component so its
  largest-magnitude entry is positive -- which makes the projection byte-for-byte
  reproducible. The reconstruction is invariant to the flip (a component and its
  coordinate column flip together), so pinning the sign changes nothing else.
- **The toy matrix is rank-2 by construction.** Only two of the four coordinates
  carry signal (royalty, gender); the other two are zero. PCA to 2-D is therefore
  *loss-less* -- it recovers the semantic plane exactly -- which is why the
  reconstruction and distance-preservation checks hold to ``1e-10`` rather than
  only approximately.
- **Cosine zero-guard.** Cosine similarity is undefined for a zero vector (its
  norm is zero). Rather than return a ``nan`` that would silently poison a
  ranking, :func:`cosine_similarity` returns ``0.0`` -- "no similarity" -- when
  either argument is the zero vector. The fixture has no zero vectors; the guard
  is defensive.
"""

from typing import NamedTuple

import numpy as np


def make_fixture() -> dict:
    """
    Build the planted toy embedding matrix every test and document shares.

    The matrix is hand-written, not trained: ``d = 4`` features by ``V = 6``
    words, with only two coordinates carrying signal -- coordinate 0 is
    *royalty*, coordinate 1 is *gender* (``+1`` masculine, ``-1`` feminine), and
    coordinates 2-3 are zero. Every similarity, neighbor, and analogy is
    therefore exact and checkable by reading the numbers, and the data lives in a
    2-D plane inside ``R^4`` so the PCA projection is loss-less.

    The vocabulary is stored sorted, which fixes the column order of ``E`` and
    the ``word_to_ix`` mapping.

    Returns
    -------
    dict
        ``{"E", "vocab", "word_to_ix", "dims"}`` where ``E`` is the ``(d, V)``
        ``float64`` embedding matrix (one word per column), ``vocab`` is the
        sorted word list, ``word_to_ix`` maps each word to its column index, and
        ``dims`` records ``{"d", "V"}``.
    """
    vocab = ["king", "man", "prince", "princess", "queen", "woman"]

    # Each word's vector as (royalty, gender, 0, 0) -- the dossier's planted axes.
    vectors = {
        "king": [2.0, 1.0, 0.0, 0.0],
        "man": [0.0, 1.0, 0.0, 0.0],
        "prince": [1.0, 1.0, 0.0, 0.0],
        "princess": [1.0, -1.0, 0.0, 0.0],
        "queen": [2.0, -1.0, 0.0, 0.0],
        "woman": [0.0, -1.0, 0.0, 0.0],
    }

    # E is (d x V): stack the word vectors as rows, then transpose so each word
    # is a column -- Ng's (d x V) convention.
    E = np.array([vectors[word] for word in vocab], dtype=np.float64).T
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    dims = {"d": E.shape[0], "V": E.shape[1]}
    return {"E": E, "vocab": vocab, "word_to_ix": word_to_ix, "dims": dims}


def one_hot(index: int, size: int) -> np.ndarray:
    """
    Build the one-hot column vector ``o_w`` for a word index.

    Parameters
    ----------
    index : int
        The word's position ``i`` in the vocabulary, ``0 <= index < size``.
    size : int
        The vocabulary size ``V`` -- the length of the returned vector.

    Returns
    -------
    np.ndarray
        A ``(size,)`` ``float64`` vector, ``1.0`` at ``index`` and ``0.0``
        elsewhere.

    Raises
    ------
    ValueError
        If ``size < 1`` or ``index`` is outside ``[0, size)``.
    """
    if size < 1:
        raise ValueError(f"size must be >= 1; got {size}")
    if not 0 <= index < size:
        raise ValueError(f"index must be in [0, {size}); got {index}")

    o = np.zeros(size, dtype=np.float64)
    o[index] = 1.0
    return o


def lookup(E: np.ndarray, index: int) -> np.ndarray:
    """
    Select a word's embedding by index -- the gather form of ``E o_w``.

    This returns the same vector as ``E @ one_hot(index, V)`` but by indexing the
    column directly, which is what every real embedding layer does: the one-hot
    matrix product is the *definition*, the column gather is the *implementation*.

    Parameters
    ----------
    E : np.ndarray
        The embedding matrix, shape ``(d, V)``.
    index : int
        The word's column index ``i``.

    Returns
    -------
    np.ndarray
        The embedding ``e_w = E[:, index]``, shape ``(d,)``.
    """
    return E[:, index]


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.

    Computes ``(u . v) / (||u|| ||v||)``, the cosine of the angle between ``u``
    and ``v``. It is ``+1`` for identical directions, ``0`` for orthogonal
    vectors, and ``-1`` for opposite directions, independent of magnitude.

    Parameters
    ----------
    u : np.ndarray
        First vector, shape ``(d,)``.
    v : np.ndarray
        Second vector, shape ``(d,)``.

    Returns
    -------
    float
        The cosine similarity, or ``0.0`` if either vector is the zero vector
        (for which the cosine is undefined -- see the module Notes).
    """
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))


def nearest_neighbors(E: np.ndarray, query: int, k: int = 1) -> list[tuple[int, float]]:
    """
    Rank the other words by cosine similarity to a query word.

    Scores every column against ``E[:, query]`` with :func:`cosine_similarity`,
    sorts by similarity descending, and returns the top ``k``. The query itself
    is excluded, and ties are broken by ascending index, so the ranking is
    deterministic.

    Parameters
    ----------
    E : np.ndarray
        The embedding matrix, shape ``(d, V)``.
    query : int
        Column index of the query word.
    k : int, optional
        How many neighbors to return. Clipped to ``V - 1`` (every word except the
        query). Defaults to ``1`` -- the single nearest neighbor.

    Returns
    -------
    list[tuple[int, float]]
        ``(index, cosine)`` pairs, most similar first, length ``min(k, V - 1)``.

    Raises
    ------
    ValueError
        If ``query`` is outside ``[0, V)`` or ``k < 1``.
    """
    V = E.shape[1]
    if not 0 <= query < V:
        raise ValueError(f"query must be in [0, {V}); got {query}")
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")

    q = E[:, query]
    scored = [(j, cosine_similarity(q, E[:, j])) for j in range(V) if j != query]

    # Descending similarity; ascending index breaks ties -> deterministic order.
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored[:k]


def analogy(E: np.ndarray, a: int, b: int, c: int) -> int:
    """
    Solve "a is to b as c is to ?" by 3CosAdd.

    Forms the target ``t = e_b - e_a + e_c`` and returns the vocabulary index
    whose embedding is most cosine-similar to ``t``. The three input words are
    excluded from the candidates -- without this, ``king - man + woman`` returns
    ``king`` -- and ties are broken by ascending index.

    The multiplicative variant 3CosMul (Levy, Goldberg & Dagan, 2015) is more
    robust on real corpora but is not implemented here; on this planted fixture
    the additive form already recovers the exact answer.

    Parameters
    ----------
    E : np.ndarray
        The embedding matrix, shape ``(d, V)``.
    a : int
        Column index of the first word (subtracted).
    b : int
        Column index of the second word (added).
    c : int
        Column index of the third word (added).

    Returns
    -------
    int
        The column index of the analogy's answer.

    Raises
    ------
    ValueError
        If any of ``a``, ``b``, ``c`` is outside ``[0, V)``, or if excluding them
        leaves no candidate words.
    """
    V = E.shape[1]
    for name, idx in (("a", a), ("b", b), ("c", c)):
        if not 0 <= idx < V:
            raise ValueError(f"{name} must be in [0, {V}); got {idx}")

    # The parallelogram target: start at c, add the b - a displacement.
    target = lookup(E, b) - lookup(E, a) + lookup(E, c)

    excluded = {a, b, c}
    best_index = -1
    best_score = -np.inf
    for j in range(V):
        if j in excluded:
            continue
        score = cosine_similarity(E[:, j], target)
        # Strict ">" keeps the first (lowest) index when two scores tie.
        if score > best_score:
            best_score = score
            best_index = j

    if best_index == -1:
        raise ValueError("no candidate words remain after excluding a, b, c")
    return best_index


class PCAProjection(NamedTuple):
    """
    The result of projecting an embedding matrix to two dimensions.

    Attributes
    ----------
    coords : np.ndarray
        The 2-D coordinates, shape ``(V, 2)`` -- one row per word, ``X_c
        W[:, :2]``.
    components : np.ndarray
        The top-2 principal directions, shape ``(2, d)``, sign-fixed for
        determinism (each row's largest-magnitude entry is made positive).
    mean : np.ndarray
        The mean word vector removed before projecting, shape ``(d,)``.
    singular_values : np.ndarray
        All singular values of the centered data, descending, shape
        ``(min(V, d),)``. The first two belong to ``components``; their squares
        are the variances explained by PC1 and PC2.
    """

    coords: np.ndarray
    components: np.ndarray
    mean: np.ndarray
    singular_values: np.ndarray


def pca_project_2d(E: np.ndarray) -> PCAProjection:
    """
    Project word vectors to two dimensions with PCA, from scratch via SVD.

    Lays the words out as rows ``X = E.T`` (``V x d``), removes the mean word
    vector to get ``X_c``, and takes the SVD ``X_c = U S W^T``. The first two rows
    of ``W^T`` are the directions of greatest variance; the 2-D coordinates are
    the centered data projected onto them, ``X_c W[:, :2]``.

    Parameters
    ----------
    E : np.ndarray
        The embedding matrix, shape ``(d, V)``.

    Returns
    -------
    PCAProjection
        The ``coords`` ``(V, 2)`` together with the ``components``, ``mean``, and
        ``singular_values`` needed to reconstruct the data and read off variance.

    Raises
    ------
    ValueError
        If ``E`` is not a 2-D array.

    Notes
    -----
    The SVD fixes each principal direction only up to sign. To make the output
    reproducible, each component is flipped so its largest-magnitude entry is
    positive; the corresponding coordinate column flips with it, so the
    reconstruction is unchanged.
    """
    if E.ndim != 2:
        raise ValueError(f"E must be 2-D (d, V); got ndim {E.ndim}")

    X = E.T  # one word per row, shape (V, d)
    mean = X.mean(axis=0)  # the mean word vector, shape (d,)
    X_c = X - mean  # center the rows on the mean word vector

    # Economy SVD: U is (V, r), S is (r,), Wt is (r, d) with r = min(V, d). We
    # project via the right singular vectors (the components) below, so the left
    # vectors U -- the word scores -- are not needed here.
    U, S, Wt = np.linalg.svd(X_c, full_matrices=False)

    components = Wt[:2].copy()  # top-2 principal directions, shape (2, d)
    # Sign convention for determinism: make each component's largest-magnitude
    # entry positive.
    for row in range(components.shape[0]):
        pivot = np.argmax(np.abs(components[row]))
        if components[row, pivot] < 0:
            components[row] = -components[row]

    coords = X_c @ components.T  # project onto the two components, shape (V, 2)
    return PCAProjection(
        coords=coords, components=components, mean=mean, singular_values=S
    )


def main() -> None:
    """Run the fixture end to end and print its known answers."""
    fixture = make_fixture()
    E, vocab, word_to_ix = fixture["E"], fixture["vocab"], fixture["word_to_ix"]
    V = E.shape[1]

    print(f"E has shape {E.shape} (d x V); vocabulary: {vocab}")

    identities = all(np.array_equal(E @ one_hot(i, V), lookup(E, i)) for i in range(V))
    print(f"E @ one_hot(i) == lookup(E, i) for every i: {identities}")

    king, queen = word_to_ix["king"], word_to_ix["queen"]
    man, woman = word_to_ix["man"], word_to_ix["woman"]
    print(f"cos(king, queen) = {cosine_similarity(E[:, king], E[:, queen]):.4f}")
    print(f"cos(man, woman)  = {cosine_similarity(E[:, man], E[:, woman]):.4f}")

    print("nearest neighbors (top-1, self excluded):")
    for i, word in enumerate(vocab):
        j, score = nearest_neighbors(E, i, k=1)[0]
        print(f"  {word:8s} -> {vocab[j]:8s} (cos = {score:.4f})")

    answer = analogy(E, a=man, b=woman, c=king)
    print(f"man : woman :: king : {vocab[answer]}")

    projection = pca_project_2d(E)
    print(
        f"PCA coords shape {projection.coords.shape}; "
        f"singular values {np.round(projection.singular_values, 4)}"
    )


if __name__ == "__main__":
    main()
