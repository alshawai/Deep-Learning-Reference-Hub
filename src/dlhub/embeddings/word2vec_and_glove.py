"""
Word2Vec and GloVe -- Learning Word Vectors from a Corpus
=========================================================

A from-scratch, teaching reference for *how* static word embeddings are
**learned**. The sibling :mod:`dlhub.embeddings.word_embeddings` treats the
vectors as *given* and reads geometry off them; this module produces those
vectors by optimising an objective over a toy corpus. It implements three
objectives, each with its loss and its exact gradient:

- **Skip-gram with negative sampling (SGNS)** -- predict context from centre,
  with ``K`` negatives drawn from the smoothed unigram sampler.
- **CBOW** -- the same tables and sampler with the roles reversed: average the
  context, predict the centre.
- **GloVe** -- a weighted least-squares fit to the log co-occurrence counts.

The gradients are the payload. Each is written term by term in the notation the
reference card uses, and :mod:`tests.test_word2vec_and_glove` checks every one
against a finite-difference approximation, because a flipped sign in a teaching
module teaches confidently.

Notation follows the dossier, which *departs* from the head topic deliberately.
The head's embedding matrix ``E`` is ``(d, V)`` (a word is a column). Here the
two learned tables are stored **row-wise**, ``(V, d)``, because every objective
indexes a word's vector by row and takes inner products -- gathering columns
from ``(d, V)`` would be all transpose noise:

- ``W`` -- the **centre** (input) table, shape ``(V, d)``; ``W[w]`` is centre
  word ``w``'s vector.
- ``C`` -- the **context** (output) table, shape ``(V, d)``; ``C[c]`` is context
  word ``c``'s vector.

The head's matrix is what this topic produces: ``E`` is the learned ``W`` (the
head's ``(d, V)`` is this module's ``W`` transposed).

References
----------
- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of
  Word Representations in Vector Space. https://arxiv.org/abs/1301.3781
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013).
  Distributed Representations of Words and Phrases and their Compositionality.
  NeurIPS. https://arxiv.org/abs/1310.4546 (Negative sampling, unigram^0.75.)
- Pennington, J., Socher, R., & Manning, C. (2014). GloVe: Global Vectors for
  Word Representation. EMNLP. https://nlp.stanford.edu/pubs/glove.pdf
- Levy, O., & Goldberg, Y. (2014). Neural Word Embedding as Implicit Matrix
  Factorization. NeurIPS. (SGNS factorises shifted PMI.)
- Rong, X. (2014). word2vec Parameter Learning Explained.
  https://arxiv.org/abs/1411.2738 (The step-by-step gradient derivation.)

Author
------
Deep Learning Reference Hub

License
-------
MIT

Notes
-----
- **Row-major tables.** ``W`` and ``C`` are ``(V, d)`` -- a word's vector is a
  *row* -- which is the opposite of the head's ``(d, V)`` ``E``. The convention
  is chosen so the objectives read as ``W[w] @ C[c]`` rather than
  ``E[:, w] @ E[:, c]``; the relationship is ``E == W.T`` after training.
- **Stable sigmoid.** :func:`sigmoid` multiplies through by ``exp(-|z|)`` so no
  intermediate ``exp`` overflows: for ``z >= 0`` it is ``1 / (1 + e^-z)`` and for
  ``z < 0`` it is ``e^z / (1 + e^z)``, which are equal to the textbook form but
  never evaluate ``exp`` of a large positive number.
- **Stable log-sigmoid in the loss.** The losses need ``log sigma(z)``.
  Evaluating ``log(sigmoid(z))`` underflows to ``-inf`` once ``sigmoid(z)``
  rounds to ``0``, so the losses use ``log sigma(z) = -softplus(-z)`` with
  ``softplus(x) = logaddexp(0, x)`` -- the same value, computed without the
  underflow. This is why the loss code calls :func:`numpy.logaddexp` rather than
  spelling the formula with ``log`` and ``sigmoid`` directly.
- **Windows do not cross sentence boundaries.** The corpus is a *list of
  sentences*; :func:`training_pairs` runs on one sentence at a time, so no
  context window straddles two sentences. On the planted fixture every sentence
  is drawn from a single semantic family, so this is exactly what makes the
  co-occurrence counts within-family and zero across families.
- **One generator drives the run.** :func:`train_sgns` seeds a single
  ``numpy.random.default_rng(seed)`` and draws from it in a fixed order -- ``W``
  then ``C`` for initialisation, then the per-pair negatives in pair order -- so
  a run is reproducible to the bit. The negatives for each pair are drawn *once*
  and reused every epoch, which fixes the objective and makes the training loss a
  clean descent curve (classic word2vec resamples every epoch; a teaching module
  that asserts a monotone loss on a toy corpus fixes the sample instead).
- **GloVe zero-count guard.** ``log X_ij`` is defined only for ``X_ij > 0``.
  :func:`glove_loss` and :func:`glove_gradients` sum over observed entries only;
  a zero count contributes no term rather than ``log 0``. GloVe's final vector is
  the sum of the two tables, ``W[i] + Wtilde[i]``.
"""

from collections.abc import Sequence

import numpy as np

from dlhub.embeddings.word_embeddings import cosine_similarity

# The planted toy corpus: four semantic families of four words each. Every
# sentence is a permutation of three words from a single family, so a context
# window never leaves the family -- co-occurrence is dense within a family and
# zero across families, which is the structure the learned neighbours recover.
_FAMILIES: dict[str, list[str]] = {
    "pets": ["cat", "dog", "fish", "bird"],
    "food": ["bread", "meat", "seed", "corn"],
    "actions": ["chase", "eat", "feed", "catch"],
    "places": ["home", "yard", "pond", "field"],
}


# Each family [a, b, c, d] contributes the same three 3-word sentences, so the
# plant is symmetric across families: the three distinct 3-subsets
# {a, b, c}, {a, b, d}, {b, c, d}. Word ``b`` is in all three (the family hub),
# and the first word ``a`` co-occurs with ``b`` in two sentences but with ``c``
# and ``d`` in one each -- so ``a``'s nearest neighbour is unambiguously ``b``,
# which is the ranking the SGNS test recovers (``cat`` -> ``dog``).
def _family_sentences(words: list[str]) -> list[list[str]]:
    a, b, c, d = words
    return [[a, b, c], [a, b, d], [b, c, d]]


def make_fixture() -> dict:
    """
    Build the planted toy corpus every test and document shares.

    The corpus is a literal list of **12 sentences x 3 tokens = 36 tokens**,
    three sentences from each of four semantic families (``pets``, ``food``,
    ``actions``, ``places``) of four words each. It is hand-written, never
    downloaded, so every frequency and co-occurrence count is known by
    construction and the sampler's distribution is closed-form.

    Returns
    -------
    dict
        ``{"corpus", "vocab", "word_to_ix", "sentences", "families", "dims"}``
        where ``corpus`` is the list of 12 word-lists, ``vocab`` is the sorted
        vocabulary (``V = 16``), ``word_to_ix`` maps each word to its index,
        ``sentences`` is ``corpus`` mapped to indices, ``families`` is the
        word-family grouping, and ``dims`` records ``{"V"}``.
    """
    corpus = [
        sentence
        for words in _FAMILIES.values()
        for sentence in _family_sentences(words)
    ]
    vocab, word_to_ix = build_vocabulary(corpus)
    sentences = [[word_to_ix[word] for word in sentence] for sentence in corpus]
    return {
        "corpus": corpus,
        "vocab": vocab,
        "word_to_ix": word_to_ix,
        "sentences": sentences,
        "families": {name: list(words) for name, words in _FAMILIES.items()},
        "dims": {"V": len(vocab)},
    }


def build_vocabulary(
    corpus: Sequence[Sequence[str]],
) -> tuple[list[str], dict[str, int]]:
    """
    Build the vocabulary and the ``word -> index`` map from a corpus.

    The vocabulary is the sorted list of unique tokens, so the index of a word is
    fixed by the corpus alone and does not depend on the order words were first
    seen. Sorting is what makes ``word_to_ix`` reproducible across runs.

    Parameters
    ----------
    corpus : Sequence[Sequence[str]]
        The corpus as a sequence of sentences, each a sequence of word strings.

    Returns
    -------
    tuple[list[str], dict[str, int]]
        ``(vocab, word_to_ix)`` -- the sorted vocabulary and the map from each
        word to its position in it.
    """
    tokens = {word for sentence in corpus for word in sentence}
    vocab = sorted(tokens)
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    return vocab, word_to_ix


def training_pairs(sentence: Sequence[int], window: int = 2) -> list[tuple[int, int]]:
    """
    Extract ``(centre, context)`` index pairs from one sentence.

    For every position ``i`` in the sentence, the centre token ``t_i`` is paired
    with each context token ``t_{i+j}`` for offset ``j`` in
    ``{-window, ..., -1, 1, ..., window}`` that lands inside the sentence. Pairs
    are emitted **by centre position, then by offset ascending**, which fixes the
    order in which stochastic descent sees them. The window is applied to a
    single sentence so it never crosses a sentence boundary.

    Parameters
    ----------
    sentence : Sequence[int]
        One sentence as a sequence of token indices.
    window : int, default=2
        The half-width ``m`` of the context window.

    Returns
    -------
    list[tuple[int, int]]
        The ``(centre, context)`` index pairs in deterministic order.

    Raises
    ------
    ValueError
        If ``window < 1``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1; got {window}")

    length = len(sentence)
    pairs: list[tuple[int, int]] = []
    for i in range(length):
        for j in range(-window, window + 1):
            if j == 0:
                continue
            k = i + j
            if 0 <= k < length:
                pairs.append((sentence[i], sentence[k]))
    return pairs


def cooccurrence_matrix(
    sentences: Sequence[Sequence[int]], vocab_size: int, window: int = 2
) -> np.ndarray:
    """
    Count co-occurrences into the ``(V, V)`` matrix GloVe fits.

    ``X[i, j]`` is the number of times word ``j`` falls inside word ``i``'s
    context window, summed over the corpus. It reuses :func:`training_pairs`, so
    the window rule is identical to the one SGNS and CBOW see, and because the
    window stays inside a sentence the counts never cross a semantic family.

    Parameters
    ----------
    sentences : Sequence[Sequence[int]]
        The corpus as sentences of token indices.
    vocab_size : int
        The vocabulary size ``V`` -- the matrix is ``(V, V)``.
    window : int, default=2
        The half-width of the context window.

    Returns
    -------
    np.ndarray
        The ``(V, V)`` integer co-occurrence matrix ``X``.
    """
    X = np.zeros((vocab_size, vocab_size), dtype=np.int64)
    for sentence in sentences:
        for centre, context in training_pairs(sentence, window):
            X[centre, context] += 1
    return X


def unigram_probabilities(
    sentences: Sequence[Sequence[int]], vocab_size: int, exponent: float = 0.75
) -> np.ndarray:
    """
    Build the smoothed unigram distribution the negative sampler draws from.

    Each word's probability is its corpus frequency raised to ``exponent`` and
    normalised::

        P(w) = f(w)^exponent / sum_w' f(w')^exponent

    The default ``exponent = 3/4`` is the Word2Vec choice. Raising a sub-unit
    power pulls the distribution toward uniform, so **rare words are drawn as
    negatives more often than their raw frequency** -- which is the point:
    negatives drawn in proportion to raw frequency would be dominated by
    stop-words and carry little signal.

    Parameters
    ----------
    sentences : Sequence[Sequence[int]]
        The corpus as sentences of token indices.
    vocab_size : int
        The vocabulary size ``V`` -- the length of the returned vector.
    exponent : float, default=0.75
        The smoothing exponent applied to each frequency.

    Returns
    -------
    np.ndarray
        A ``(V,)`` ``float64`` probability vector that sums to 1.
    """
    counts = np.zeros(vocab_size, dtype=np.float64)
    for sentence in sentences:
        for token in sentence:
            counts[token] += 1.0

    smoothed = counts**exponent
    return smoothed / smoothed.sum()


def sample_negatives(
    rng: np.random.Generator, probabilities: np.ndarray, k: int
) -> np.ndarray:
    """
    Draw ``k`` negative word indices i.i.d. from the unigram sampler.

    This is ``rng.choice(V, size=k, p=probabilities)`` -- a thin wrapper that
    names the operation and keeps every draw on the single generator the training
    run threads through, so the whole run stays reproducible.

    Parameters
    ----------
    rng : np.random.Generator
        The run's generator; advancing it here is what keeps the draw order
        deterministic.
    probabilities : np.ndarray
        The ``(V,)`` sampling distribution from :func:`unigram_probabilities`.
    k : int
        How many negatives to draw.

    Returns
    -------
    np.ndarray
        A ``(k,)`` array of word indices.
    """
    vocab_size = probabilities.shape[0]
    return rng.choice(vocab_size, size=k, p=probabilities)


def sigmoid(z: float | np.ndarray) -> float | np.ndarray:
    """
    The logistic sigmoid ``1 / (1 + e^-z)``, computed without overflow.

    The textbook spelling evaluates ``exp(-z)``, which overflows for large
    negative ``z``. Multiplying through by ``exp(-|z|)`` -- which never exceeds
    ``1`` -- gives the identical value from a bounded exponential: ``1 / (1 +
    e^-z)`` for ``z >= 0`` and ``e^z / (1 + e^z)`` for ``z < 0``.

    Parameters
    ----------
    z : float or np.ndarray
        The pre-activation score (a scalar inner product, or an array of them).

    Returns
    -------
    float or np.ndarray
        ``sigma(z)`` in ``(0, 1)``, a ``float`` for scalar input.
    """
    z = np.asarray(z, dtype=np.float64)
    exp_neg_abs = np.exp(-np.abs(z))
    result = np.where(
        z >= 0.0, 1.0 / (1.0 + exp_neg_abs), exp_neg_abs / (1.0 + exp_neg_abs)
    )
    return float(result) if result.ndim == 0 else result


def _log_sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """
    ``log sigma(z)``, computed as ``-softplus(-z)`` so it never underflows.

    ``log(sigmoid(z))`` rounds to ``-inf`` once ``sigmoid(z)`` underflows to
    ``0``. The identity ``log sigma(z) = -log(1 + e^-z) = -softplus(-z)`` with
    ``softplus(x) = logaddexp(0, x)`` returns the same number from a stable path.
    """
    return -np.logaddexp(0.0, -np.asarray(z, dtype=np.float64))


def sgns_loss(
    W: np.ndarray,
    C: np.ndarray,
    centre: int,
    context: int,
    negatives: Sequence[int],
) -> float:
    """
    Skip-gram negative-sampling loss for one ``(centre, context)`` pair.

    With centre vector ``v = W[centre]``, positive context ``u = C[context]`` and
    ``K`` negatives ``u_k = C[n_k]``::

        l = -log sigma(v . u) - sum_k log sigma(-v . u_k)

    The positive term rewards a high score for the true context; each negative
    term rewards a *low* score for a sampled non-context. The ``sigma(-z)``
    spelling of the negative term is the numerically stable one (see the module
    Notes), and both terms are evaluated through :func:`_log_sigmoid`.

    Parameters
    ----------
    W : np.ndarray
        The centre table, shape ``(V, d)``.
    C : np.ndarray
        The context table, shape ``(V, d)``.
    centre : int
        Index of the centre word.
    context : int
        Index of the true context word.
    negatives : Sequence[int]
        The ``K`` sampled negative indices.

    Returns
    -------
    float
        The per-pair loss.
    """
    v = W[centre]
    u = C[context]
    positive = _log_sigmoid(v @ u)
    negative = sum(_log_sigmoid(-(v @ C[n])) for n in negatives)
    return float(-positive - negative)


def sgns_gradients(
    W: np.ndarray,
    C: np.ndarray,
    centre: int,
    context: int,
    negatives: Sequence[int],
) -> dict[str, np.ndarray]:
    """
    Gradients of :func:`sgns_loss` with respect to both tables.

    Term by term, with ``v = W[centre]``, ``u = C[context]`` and
    ``u_k = C[n_k]``::

        dl/dv       = (sigma(v . u) - 1) u + sum_k sigma(v . u_k) u_k
        dl/du       = (sigma(v . u) - 1) v
        dl/du_k     =  sigma(v . u_k) v

    Only the rows a pair touches receive a gradient, so the returned tables are
    zero except at ``centre`` (in ``W``) and at ``context`` and the negatives (in
    ``C``). Contributions are accumulated with ``+=`` one index at a time, so a
    negative that happens to equal the context -- or a repeated negative -- adds
    correctly rather than overwriting.

    Parameters
    ----------
    W : np.ndarray
        The centre table, shape ``(V, d)``.
    C : np.ndarray
        The context table, shape ``(V, d)``.
    centre : int
        Index of the centre word.
    context : int
        Index of the true context word.
    negatives : Sequence[int]
        The ``K`` sampled negative indices.

    Returns
    -------
    dict[str, np.ndarray]
        ``{"W": dW, "C": dC}`` -- full ``(V, d)`` gradient tables, zero off the
        touched rows.
    """
    v = W[centre]
    u = C[context]

    dW = np.zeros_like(W)
    dC = np.zeros_like(C)

    # Positive pair: pull the centre and its true context together.
    sigma_pos = sigmoid(v @ u)
    dW[centre] += (sigma_pos - 1.0) * u
    dC[context] += (sigma_pos - 1.0) * v

    # Negatives: push the centre away from each sampled non-context.
    for n in negatives:
        u_n = C[n]
        sigma_neg = sigmoid(v @ u_n)
        dW[centre] += sigma_neg * u_n
        dC[n] += sigma_neg * v

    return {"W": dW, "C": dC}


def cbow_loss(
    W: np.ndarray,
    C: np.ndarray,
    centre: int,
    context_words: Sequence[int],
    negatives: Sequence[int],
) -> float:
    """
    CBOW negative-sampling loss for one ``(context -> centre)`` example.

    CBOW reverses skip-gram: average the context vectors and predict the centre.
    With ``ubar = mean_j C[c_j]`` over the context words, centre vector
    ``v = W[centre]`` and negatives scored against the **centre** table,
    ``v_k = W[n_k]``::

        l = -log sigma(ubar . v) - sum_k log sigma(-ubar . v_k)

    It is :func:`sgns_loss` with ``ubar`` in the centre's place and the
    centre/negative words looked up in ``W`` -- the same objective, the scoring
    swapped.

    Parameters
    ----------
    W : np.ndarray
        The centre table, shape ``(V, d)``.
    C : np.ndarray
        The context table, shape ``(V, d)``.
    centre : int
        Index of the centre word being predicted.
    context_words : Sequence[int]
        The context word indices whose vectors are averaged.
    negatives : Sequence[int]
        The ``K`` sampled negative centre indices.

    Returns
    -------
    float
        The per-example loss.
    """
    u_bar = C[list(context_words)].mean(axis=0)
    v = W[centre]
    positive = _log_sigmoid(u_bar @ v)
    negative = sum(_log_sigmoid(-(u_bar @ W[n])) for n in negatives)
    return float(-positive - negative)


def cbow_gradients(
    W: np.ndarray,
    C: np.ndarray,
    centre: int,
    context_words: Sequence[int],
    negatives: Sequence[int],
) -> dict[str, np.ndarray]:
    """
    Gradients of :func:`cbow_loss` with respect to both tables.

    With ``ubar = mean_j C[c_j]`` and ``M = len(context_words)``::

        dl/dubar  = (sigma(ubar . v) - 1) v + sum_k sigma(ubar . v_k) v_k
        dl/dW[w]  = (sigma(ubar . v) - 1) ubar
        dl/dW[n_k]=  sigma(ubar . v_k) ubar
        dl/dC[c_j]= (1 / M) dl/dubar     for each context word c_j

    The averaging splits the gradient on ``ubar`` evenly back to each context
    row (``1 / M``); a word that appears twice in the window collects the share
    twice, which the one-index-at-a-time ``+=`` handles.

    Parameters
    ----------
    W : np.ndarray
        The centre table, shape ``(V, d)``.
    C : np.ndarray
        The context table, shape ``(V, d)``.
    centre : int
        Index of the centre word being predicted.
    context_words : Sequence[int]
        The context word indices whose vectors are averaged.
    negatives : Sequence[int]
        The ``K`` sampled negative centre indices.

    Returns
    -------
    dict[str, np.ndarray]
        ``{"W": dW, "C": dC}`` -- full ``(V, d)`` gradient tables.
    """
    context_list = list(context_words)
    count = len(context_list)
    u_bar = C[context_list].mean(axis=0)
    v = W[centre]

    dW = np.zeros_like(W)
    dC = np.zeros_like(C)

    # Gradient flowing into the averaged context vector ubar.
    sigma_pos = sigmoid(u_bar @ v)
    grad_u_bar = (sigma_pos - 1.0) * v
    dW[centre] += (sigma_pos - 1.0) * u_bar

    for n in negatives:
        v_n = W[n]
        sigma_neg = sigmoid(u_bar @ v_n)
        grad_u_bar = grad_u_bar + sigma_neg * v_n
        dW[n] += sigma_neg * u_bar

    # Split the gradient on ubar evenly across the context rows that formed it.
    for c in context_list:
        dC[c] += grad_u_bar / count

    return {"W": dW, "C": dC}


def glove_weight(
    x: np.ndarray | float, x_max: float = 100.0, alpha: float = 0.75
) -> np.ndarray | float:
    """
    GloVe's co-occurrence weighting ``f(x) = min(1, (x / x_max)^alpha)``.

    The weight grows with the count up to ``x_max`` and is capped at ``1``
    thereafter, so very frequent pairs do not dominate the fit while rare pairs
    are not ignored. It is only ever evaluated at observed counts ``x > 0``.

    Parameters
    ----------
    x : np.ndarray or float
        A co-occurrence count (or array of them).
    x_max : float, default=100.0
        The count at which the weight saturates to ``1``.
    alpha : float, default=0.75
        The exponent controlling the ramp below ``x_max``.

    Returns
    -------
    np.ndarray or float
        The weight ``f(x)`` in ``(0, 1]``.
    """
    return np.minimum(1.0, (np.asarray(x, dtype=np.float64) / x_max) ** alpha)


def glove_loss(
    W: np.ndarray,
    W_tilde: np.ndarray,
    b: np.ndarray,
    b_tilde: np.ndarray,
    X: np.ndarray,
    x_max: float = 100.0,
    alpha: float = 0.75,
) -> float:
    """
    GloVe weighted least-squares loss over the observed co-occurrences.

    For every ``X[i, j] > 0`` the residual is
    ``r_ij = W[i] . W_tilde[j] + b[i] + b_tilde[j] - log X[i, j]`` and::

        J = sum_{i,j: X_ij > 0} f(X_ij) * r_ij^2

    Zero counts contribute no term -- GloVe fits ``log X`` only where ``X`` is
    observed, which is also what keeps ``log 0`` out of the sum.

    Parameters
    ----------
    W : np.ndarray
        The word table, shape ``(V, d)``.
    W_tilde : np.ndarray
        The context-word table, shape ``(V, d)``.
    b : np.ndarray
        The word biases, shape ``(V,)``.
    b_tilde : np.ndarray
        The context-word biases, shape ``(V,)``.
    X : np.ndarray
        The ``(V, V)`` co-occurrence matrix.
    x_max : float, default=100.0
        The weighting cap passed to :func:`glove_weight`.
    alpha : float, default=0.75
        The weighting exponent passed to :func:`glove_weight`.

    Returns
    -------
    float
        The total loss ``J``.
    """
    rows, cols = np.nonzero(X)
    total = 0.0
    for i, j in zip(rows, cols):
        residual = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(X[i, j])
        total += glove_weight(X[i, j], x_max, alpha) * residual**2
    return float(total)


def glove_gradients(
    W: np.ndarray,
    W_tilde: np.ndarray,
    b: np.ndarray,
    b_tilde: np.ndarray,
    X: np.ndarray,
    x_max: float = 100.0,
    alpha: float = 0.75,
) -> dict[str, np.ndarray]:
    """
    Gradients of :func:`glove_loss` with respect to tables and biases.

    With ``r_ij`` and ``f(X_ij)`` as in :func:`glove_loss`, each observed entry
    contributes::

        dJ/dW[i]       += 2 f(X_ij) r_ij W_tilde[j]
        dJ/dW_tilde[j] += 2 f(X_ij) r_ij W[i]
        dJ/db[i]       += 2 f(X_ij) r_ij
        dJ/db_tilde[j] += 2 f(X_ij) r_ij

    Parameters
    ----------
    W : np.ndarray
        The word table, shape ``(V, d)``.
    W_tilde : np.ndarray
        The context-word table, shape ``(V, d)``.
    b : np.ndarray
        The word biases, shape ``(V,)``.
    b_tilde : np.ndarray
        The context-word biases, shape ``(V,)``.
    X : np.ndarray
        The ``(V, V)`` co-occurrence matrix.
    x_max : float, default=100.0
        The weighting cap passed to :func:`glove_weight`.
    alpha : float, default=0.75
        The weighting exponent passed to :func:`glove_weight`.

    Returns
    -------
    dict[str, np.ndarray]
        ``{"W", "Wt", "b", "bt"}`` -- gradients matching each parameter's shape.
    """
    dW = np.zeros_like(W)
    dW_tilde = np.zeros_like(W_tilde)
    db = np.zeros_like(b)
    db_tilde = np.zeros_like(b_tilde)

    rows, cols = np.nonzero(X)
    for i, j in zip(rows, cols):
        residual = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(X[i, j])
        scale = 2.0 * glove_weight(X[i, j], x_max, alpha) * residual
        dW[i] += scale * W_tilde[j]
        dW_tilde[j] += scale * W[i]
        db[i] += scale
        db_tilde[j] += scale

    return {"W": dW, "Wt": dW_tilde, "b": db, "bt": db_tilde}


def _init_table(rng: np.random.Generator, vocab_size: int, d: int) -> np.ndarray:
    """Draw a ``(V, d)`` table from ``N(0, sigma^2)`` with ``sigma = 0.1 / sqrt(d)``.

    A tiny *non-zero* initialisation is deliberate: two identical rows would
    receive identical gradients forever and never separate, so the tables must
    start apart. The ``1 / sqrt(d)`` scale keeps the initial inner products
    ``O(0.01)`` regardless of the dimension.
    """
    sigma = 0.1 / np.sqrt(d)
    return rng.normal(0.0, sigma, size=(vocab_size, d))


def train_sgns(
    sentences: Sequence[Sequence[int]],
    vocab_size: int,
    *,
    d: int = 8,
    window: int = 2,
    k: int = 2,
    lr: float = 0.05,
    epochs: int = 200,
    seed: int = 0,
    exponent: float = 0.75,
) -> dict:
    """
    Train skip-gram with negative sampling by full-batch gradient descent.

    A single seeded generator draws the two tables (``W`` then ``C``) and then
    the ``k`` negatives for every ``(centre, context)`` pair, in pair order. The
    negatives are drawn **once** and reused each epoch, so the objective is fixed
    and the loss is a clean descent curve. Each epoch sums the loss and the
    gradient over all pairs and takes one plain SGD step -- no optimiser, no
    momentum (that lesson lives in ``optimizer-update-rules``).

    Parameters
    ----------
    sentences : Sequence[Sequence[int]]
        The corpus as sentences of token indices.
    vocab_size : int
        The vocabulary size ``V``.
    d : int, default=8
        The embedding dimension.
    window : int, default=2
        The context half-width.
    k : int, default=2
        The number of negatives per pair.
    lr : float, default=0.05
        The gradient-descent step size.
    epochs : int, default=200
        The number of full-batch passes.
    seed : int, default=0
        Seeds the single generator driving init and sampling.
    exponent : float, default=0.75
        The negative sampler's smoothing exponent.

    Returns
    -------
    dict
        ``{"W", "C", "history"}`` -- the trained ``(V, d)`` tables and the list
        of per-epoch total losses (each recorded *before* that epoch's update,
        so it is strictly decreasing on this fixture).
    """
    rng = np.random.default_rng(seed)
    W = _init_table(rng, vocab_size, d)
    C = _init_table(rng, vocab_size, d)

    probabilities = unigram_probabilities(sentences, vocab_size, exponent)
    pairs = [
        pair for sentence in sentences for pair in training_pairs(sentence, window)
    ]
    negatives_per_pair = [sample_negatives(rng, probabilities, k) for _ in pairs]

    history: list[float] = []
    for _ in range(epochs):
        total_loss = 0.0
        dW = np.zeros_like(W)
        dC = np.zeros_like(C)
        for (centre, context), negatives in zip(pairs, negatives_per_pair):
            total_loss += sgns_loss(W, C, centre, context, negatives)
            grads = sgns_gradients(W, C, centre, context, negatives)
            dW += grads["W"]
            dC += grads["C"]

        history.append(total_loss)
        W -= lr * dW
        C -= lr * dC

    return {"W": W, "C": C, "history": history}


def train_glove(
    X: np.ndarray,
    vocab_size: int,
    *,
    d: int = 8,
    lr: float = 0.05,
    epochs: int = 200,
    seed: int = 0,
    x_max: float = 100.0,
    alpha: float = 0.75,
) -> dict:
    """
    Fit GloVe to a co-occurrence matrix by full-batch gradient descent.

    A single seeded generator draws the two tables and the two bias vectors in
    the order ``W``, ``W_tilde``, ``b``, ``b_tilde``. Each epoch sums the loss
    and gradient over the observed entries of ``X`` and takes one SGD step. The
    learned vector for a word is the **sum of the two tables**,
    ``W[i] + W_tilde[i]``, returned as ``vectors``.

    Parameters
    ----------
    X : np.ndarray
        The ``(V, V)`` co-occurrence matrix from :func:`cooccurrence_matrix`.
    vocab_size : int
        The vocabulary size ``V``.
    d : int, default=8
        The embedding dimension.
    lr : float, default=0.05
        The gradient-descent step size.
    epochs : int, default=200
        The number of full-batch passes.
    seed : int, default=0
        Seeds the single generator driving initialisation.
    x_max : float, default=100.0
        The weighting cap passed to :func:`glove_weight`.
    alpha : float, default=0.75
        The weighting exponent passed to :func:`glove_weight`.

    Returns
    -------
    dict
        ``{"W", "Wt", "b", "bt", "vectors", "history"}`` -- the trained tables
        and biases, the summed word vectors ``(V, d)``, and the per-epoch loss.
    """
    rng = np.random.default_rng(seed)
    W = _init_table(rng, vocab_size, d)
    W_tilde = _init_table(rng, vocab_size, d)
    sigma = 0.1 / np.sqrt(d)
    b = rng.normal(0.0, sigma, size=vocab_size)
    b_tilde = rng.normal(0.0, sigma, size=vocab_size)

    history: list[float] = []
    for _ in range(epochs):
        history.append(glove_loss(W, W_tilde, b, b_tilde, X, x_max, alpha))
        grads = glove_gradients(W, W_tilde, b, b_tilde, X, x_max, alpha)
        W -= lr * grads["W"]
        W_tilde -= lr * grads["Wt"]
        b -= lr * grads["b"]
        b_tilde -= lr * grads["bt"]

    return {
        "W": W,
        "Wt": W_tilde,
        "b": b,
        "bt": b_tilde,
        "vectors": W + W_tilde,
        "history": history,
    }


def nearest_neighbor(embeddings: np.ndarray, query: int, k: int = 1) -> list[int]:
    """
    Rank the ``k`` nearest words to a query by cosine similarity.

    Scores every row of ``embeddings`` against ``embeddings[query]`` with the
    head topic's :func:`~dlhub.embeddings.word_embeddings.cosine_similarity` --
    this topic *produces* vectors, it does not re-derive their geometry -- then
    returns the top ``k`` indices, the query excluded, ties broken by ascending
    index so the ranking is deterministic.

    Parameters
    ----------
    embeddings : np.ndarray
        A ``(V, d)`` table whose rows are word vectors (e.g. the trained ``W``).
    query : int
        The row index of the query word.
    k : int, default=1
        How many neighbours to return.

    Returns
    -------
    list[int]
        The ``k`` nearest word indices, most similar first.

    Raises
    ------
    ValueError
        If ``query`` is out of range or ``k < 1``.
    """
    vocab_size = embeddings.shape[0]
    if not 0 <= query < vocab_size:
        raise ValueError(f"query must be in [0, {vocab_size}); got {query}")
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")

    q = embeddings[query]
    scored = [
        (j, cosine_similarity(q, embeddings[j]))
        for j in range(vocab_size)
        if j != query
    ]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return [j for j, _ in scored[:k]]


def main() -> None:
    """Train SGNS on the fixture and print the pets family's recovered neighbours."""
    fixture = make_fixture()
    sentences, vocab, word_to_ix = (
        fixture["sentences"],
        fixture["vocab"],
        fixture["word_to_ix"],
    )
    V = fixture["dims"]["V"]

    result = train_sgns(sentences, V)
    history = result["history"]
    print(
        f"V = {V}; SGNS loss {history[0]:.3f} -> {history[-1]:.3f} over {len(history)} epochs"
    )

    cat = word_to_ix["cat"]
    neighbours = nearest_neighbor(result["W"], cat, k=3)
    print(f"nearest to 'cat': {[vocab[j] for j in neighbours]}")


if __name__ == "__main__":
    main()
