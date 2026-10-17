"""
Debiasing Word Embeddings -- Bias Geometry, Hard Debiasing, and WEAT
====================================================================

A from-scratch, teaching reference for the **geometry of bias** in a given
static embedding: how gender bias lives in a *direction* of the vector space,
how Bolukbasi et al.'s hard debiasing removes it (identify the subspace,
neutralize, equalize), and how WEAT (Caliskan et al.) measures the association
that remains -- used here to quantify bias *before and after* debiasing.

The vectors are treated as **given** -- a small, hand-planted toy matrix (see
:func:`make_fixture`) whose every answer is exact and checkable by eye. What a
word vector *is* belongs to the sibling :mod:`dlhub.embeddings.word_embeddings`;
how vectors are *learned* belongs to
:mod:`dlhub.embeddings.word2vec_and_glove`. Cosine similarity and nearest-
neighbor ranking are imported from the head rather than restated: this module
studies only the bias geometry on top of them.

The module's thesis is cautionary. After :func:`hard_debias`, every direct
projection on the gender direction is zero and WEAT's effect size ``d`` is
exactly ``0`` -- the metrics say "fixed". Yet the neutral-neighbour map
(:func:`neutral_neighbors`) is *identical* before and after: the words still
group by gender through the indirect cluster axis the debiasing never touched.
That is Gonen & Goldberg's "lipstick on a pig" critique, demonstrated in a
four-dimensional space small enough to read by eye: geometric debiasing is
superficial.

Layout is ``(V, d)`` -- one word per row, so a word's vector is ``E[i]``. This
is the head's ``E`` transposed (``E == head_E.T``), the convention the dossier
settles because projections, WEAT, and neighbour structure are transpose-free
on rows; where the head's column-convention helpers are reused, the row matrix
is passed transposed (see :func:`neutral_neighbors`). Every row is
**unit-norm** -- :func:`make_fixture` L2-normalizes, which :func:`equalize`
requires.

References
----------
- Bolukbasi, T., Chang, K.-W., Zou, J., Saligrama, V., & Kalai, A. (2016). Man
  is to Computer Programmer as Woman is to Homemaker? Debiasing Word
  Embeddings. NeurIPS. https://arxiv.org/abs/1607.06520
- Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). Semantics derived
  automatically from language corpora contain human-like biases. Science,
  356(6334), 183-186. (WEAT.)
- Gonen, H., & Goldberg, Y. (2019). Lipstick on a Pig: Debiasing Methods Cover
  up Systematic Gender Biases. NAACL. https://arxiv.org/abs/1903.03862
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
- **Deterministic male-positive sign.** The SVD fixes the gender direction only
  up to sign, so :func:`gender_direction` flips ``g`` when needed to make
  ``g . (e_he - e_she) > 0`` for the *first* definitional pair -- the same
  determinism-by-convention the head's ``pca_project_2d`` uses. Without it, two
  runs could report mirrored projections and a WEAT statistic with a flipped
  sign.
- **Stacked-deviation SVD = Bolukbasi's covariance PCA.** The paper's Step 1
  takes the SVD of the covariance
  ``C = sum_i sum_w (w - mu_i)^T (w - mu_i) / |D_i|`` over the defining sets;
  this module instead stacks the deviations ``w - mu_i`` themselves and takes
  their SVD. The right singular vectors of the stacked deviations are the
  eigenvectors of its Gram matrix, so the two forms give the same principal
  directions, and with definitional *pairs* the ``1/|D_i|`` weight is the
  uniform ``1/2``, so they coincide exactly. The stacked form is the one shown
  because it displays the data the SVD consumes.
- **A single direction (k = 1).** Bolukbasi generalizes to a k-dimensional
  subspace ``B`` (they use ``k = 10`` on real corpora), with the projection
  ``v_B = sum_j (v . b_j) b_j``. The fixture's deviation matrix is exactly
  rank-1 -- singular values ``[sqrt(6), 0, 0, 0]`` -- so one direction *is*
  the whole subspace here, and every function takes a single unit vector ``g``.
- **Normalize first.** Equalize's ``sqrt(1 - ||nu||^2)`` is real only when the
  set members are unit vectors (then ``||nu|| <= 1``, ``nu`` being the neutral
  part of their mean). :func:`make_fixture` L2-normalizes every row, and
  :func:`equalize` raises rather than return a ``nan`` when ``||nu|| > 1``.
- **Neutralize zero-residual guard.** A word that lies entirely along ``g``
  (like the fixture's definitional ``he``) has no neutral part: the residual
  ``w - (w.g) g`` is the zero vector and renormalizing would divide by zero.
  :func:`neutralize` returns the zero vector instead, which the head's cosine
  zero-guard then keeps out of every ranking. :func:`hard_debias` never calls
  it on definitional words -- those define ``g`` and are left untouched, as in
  Bolukbasi.
- **Population standard deviation in WEAT's effect size.** ``d`` divides by the
  *population* std (``ddof = 0``) of the pooled association scores over
  ``X u Y``; under the sample std (``ddof = 1``) the fixture's validated answer
  would be 1.8102397229 instead of 1.9830182615.
- **Exact permutation p.** :func:`weat_pvalue` enumerates *every*
  ``C(|X u Y|, |X|)`` equal-size partition -- 20 for the fixture's 3-vs-3
  targets -- and reports the fraction whose statistic is ``>=`` the observed,
  so the observed partition always counts itself and ``p >= 1/C``. Exact
  enumeration is feasible only because the toy sets are tiny; Caliskan et al.
  sample 10,000 random permutations for real-size sets.
- **The d zero-spread guard.** When every pooled score is equal the std is
  zero and the effect size is ``0/0``; :func:`weat_effect_size` returns ``0.0``
  by definition rather than a ``nan``. That is exactly the post-debias case --
  every neutralized word scores ``0`` against the pure-gender attributes --
  and the dossier pins ``d_after = 0`` to this guard.
"""

import itertools
from collections.abc import Sequence

import numpy as np

from dlhub.embeddings.word_embeddings import cosine_similarity, nearest_neighbors


def make_fixture() -> dict:
    """
    Build the planted bias fixture every test and document shares.

    The matrix is hand-written, not trained: ``V = 14`` words by ``d = 4``
    coordinates, committed as an integer literal and L2-normalized row-wise so
    every row is unit-norm (normalize-first, as Bolukbasi requires). The
    coordinates are ``0`` = gender (the bias axis), ``1`` = occupation cluster
    (the *indirect*-bias axis, correlated with gender -- this is what survives
    neutralize), ``2-3`` = within-cluster content (tie-free neighbours).

    The six definitional words are pure-gender archetypes and coincide within
    gender (``he = man = boy = +e0``, ``she = woman = girl = -e0``): their only
    role is to define ``g`` and to serve as WEAT's gender attributes, and
    nothing computes neighbours among them, so the coincidence keeps every
    known answer exact without affecting any result.

    The vocabulary is stored sorted, which fixes the row order of ``E`` and the
    ``word_to_ix`` mapping.

    Returns
    -------
    dict
        ``{"E", "vocab", "word_to_ix", "dims", "definitional_pairs",
        "neutral_words", "equality_sets", "weat_sets"}`` where ``E`` is the
        ``(V, d)`` ``float64`` matrix with unit-norm rows (one word per row,
        the head's ``E.T``), ``vocab`` the sorted word list, ``word_to_ix`` the
        word-to-row map, ``dims`` the ``{"V", "d"}`` sizes, and the four role
        sets named by word: the definitional ``(male, female)`` pairs, the
        gender-neutral words to neutralize, the equality sets to equalize, and
        the WEAT ``A``/``B`` attribute and ``X``/``Y`` target sets.
    """
    # Each word's raw vector as (gender, cluster, within_a, within_b) -- the
    # dossier's planted axes, as integers so every answer is checkable by eye.
    vectors = {
        "boy": [1, 0, 0, 0],
        "engineer": [1, 3, 3, 0],
        "girl": [-1, 0, 0, 0],
        "grandfather": [2, 0, 0, 1],
        "grandmother": [-1, 0, 0, 1],
        "he": [1, 0, 0, 0],
        "man": [1, 0, 0, 0],
        "mechanic": [1, 3, 0, 0],
        "nanny": [-1, -3, 0, 0],
        "nurse": [-1, -3, 3, 0],
        "programmer": [1, 3, 2, 0],
        "receptionist": [-1, -3, 2, 0],
        "she": [-1, 0, 0, 0],
        "woman": [-1, 0, 0, 0],
    }

    vocab = sorted(vectors)
    raw = np.array([vectors[word] for word in vocab], dtype=np.float64)
    # Normalize-first: equalize's sqrt(1 - ||nu||^2) is only real on the unit
    # sphere, so every row is L2-normalized to length 1.
    E = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    word_to_ix = {word: i for i, word in enumerate(vocab)}

    return {
        "E": E,
        "vocab": vocab,
        "word_to_ix": word_to_ix,
        "dims": {"V": E.shape[0], "d": E.shape[1]},
        "definitional_pairs": [("he", "she"), ("man", "woman"), ("boy", "girl")],
        "neutral_words": [
            "engineer",
            "programmer",
            "mechanic",
            "nurse",
            "receptionist",
            "nanny",
        ],
        "equality_sets": [["grandfather", "grandmother"]],
        "weat_sets": {
            "A": ["he", "man", "boy"],
            "B": ["she", "woman", "girl"],
            "X": ["engineer", "programmer", "mechanic"],
            "Y": ["nurse", "receptionist", "nanny"],
        },
    }


def _rows(
    E: np.ndarray, word_to_ix: dict[str, int], words: Sequence[str]
) -> np.ndarray:
    """Gather the named words' vectors into a ``(len(words), d)`` row matrix."""
    return E[[word_to_ix[word] for word in words]]


def gender_direction(
    E: np.ndarray, definitional_pairs: Sequence[tuple[int, int]]
) -> np.ndarray:
    """
    Identify the unit gender direction ``g`` from definitional pairs.

    For each ``(male, female)`` pair, take the pair mean ``mu`` and the two
    deviations ``w - mu``; stack all deviations and take the **top principal
    component** (via SVD) as ``g`` (Bolukbasi §5.1, in the stacked-deviation
    form -- see the module Notes). The SVD fixes a direction only up to sign,
    so the sign is pinned deterministically **male-positive**: flip ``g`` when
    ``g . (e_male - e_female) < 0`` for the first pair.

    A single direction (``k = 1``) is returned; the method generalizes to a
    ``k``-dimensional subspace ``B`` by taking the top ``k`` right singular
    vectors instead.

    Parameters
    ----------
    E : np.ndarray
        The embedding matrix, shape ``(V, d)``, one word per row.
    definitional_pairs : Sequence[tuple[int, int]]
        Row indices ``(male, female)`` of words whose definitions differ only
        in gender -- e.g. ``(he, she)``, ``(man, woman)``, ``(boy, girl)``.

    Returns
    -------
    np.ndarray
        The unit gender direction ``g``, shape ``(d,)``, male-positive.

    Raises
    ------
    ValueError
        If ``definitional_pairs`` is empty.
    """
    if len(definitional_pairs) == 0:
        raise ValueError("definitional_pairs must hold at least one pair")

    # Center each definitional pair on its own mean and stack the deviations:
    # one row per word, shape (2 * n_pairs, d).
    deviations = []
    for male, female in definitional_pairs:
        mu = (E[male] + E[female]) / 2.0
        deviations.append(E[male] - mu)
        deviations.append(E[female] - mu)
    deviation_matrix = np.array(deviations)

    # The top principal component is the first right singular vector. (The
    # singular values themselves show why k = 1 suffices on the fixture: the
    # spectrum is [sqrt(6), 0, 0, 0], exactly rank-1.)
    _, _, Wt = np.linalg.svd(deviation_matrix, full_matrices=False)
    g = Wt[0].copy()

    # Deterministic sign rule: the male direction is positive.
    male, female = definitional_pairs[0]
    if np.dot(g, E[male] - E[female]) < 0.0:
        g = -g
    return g


def neutralize(w: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Remove a gender-neutral word's ``g`` component and renormalize.

    The projection out of the gender direction, spelled term by term::

        w' = (w - (w . g) g) / || w - (w . g) g ||

    After this, ``w' . g = 0`` (the direct bias is gone) and ``w'`` is back on
    the unit sphere, so cosines and distances stay comparable.

    Parameters
    ----------
    w : np.ndarray
        The word's vector, shape ``(d,)``.
    g : np.ndarray
        The unit gender direction, shape ``(d,)``.

    Returns
    -------
    np.ndarray
        The neutralized unit vector, shape ``(d,)`` -- or the zero vector when
        ``w`` lies entirely along ``g`` and has no neutral part (the defensive
        guard; see the module Notes).
    """
    residual = w - np.dot(w, g) * g
    norm = float(np.linalg.norm(residual))
    if norm == 0.0:
        # w is pure-gender: nothing survives the projection. Return the zero
        # vector rather than divide by zero; the head's cosine zero-guard keeps
        # it out of every ranking.
        return np.zeros_like(residual)
    return residual / norm


def equalize(vectors: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Equalize one equality set: make its members symmetric about ``g``.

    For a gendered pair to keep *distinct* but unbiased (``{grandfather,
    grandmother}``), Bolukbasi's Step 2a (arXiv:1607.06520) replaces every
    member's neutral part with the set's shared neutral part ``nu`` and gives
    the members equal-and-opposite ``g`` components::

        mu   = mean(E)                  # the set mean
        mu_B = (mu . g) g               # its gender component
        nu   = mu - mu_B                # the shared neutral part
        w_B  = (w . g) g                # each member's gender component

        w'   = nu + sqrt(1 - ||nu||^2) * (w_B - mu_B) / || w_B - mu_B ||

    Every output is unit-norm, the members' ``g`` projections are ``+/-`` the
    shared ``sqrt(1 - ||nu||^2)``, and all members are equidistant from any
    neutralized word -- which is what makes the pair unbiased without merging
    the two words into one vector.

    Parameters
    ----------
    vectors : np.ndarray
        The equality set's member vectors, shape ``(n, d)``, **unit-norm**
        (normalize-first: ``sqrt(1 - ||nu||^2)`` is only real on the unit
        sphere).
    g : np.ndarray
        The unit gender direction, shape ``(d,)``.

    Returns
    -------
    np.ndarray
        The equalized vectors, shape ``(n, d)``, in the input's member order.

    Raises
    ------
    ValueError
        If ``||nu|| > 1`` (the members are not unit-norm), or if some member's
        gender component equals the set mean's, making ``w_B - mu_B`` zero and
        the direction undefined.
    """
    mu = vectors.mean(axis=0)
    mu_B = np.dot(mu, g) * g
    nu = mu - mu_B

    # The shared g-component magnitude. Real only on the unit sphere.
    neutral_sq = float(np.dot(nu, nu))
    if neutral_sq > 1.0:
        raise ValueError(
            "equalize needs unit-norm members: ||nu||^2 = "
            f"{neutral_sq:.6f} > 1, so sqrt(1 - ||nu||^2) is not real "
            "(normalize the rows first)"
        )
    scale = np.sqrt(1.0 - neutral_sq)

    equalized = np.empty_like(vectors)
    for row in range(vectors.shape[0]):
        w_B = np.dot(vectors[row], g) * g
        direction = w_B - mu_B
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            raise ValueError(
                f"member {row} has the set mean's gender component, so the "
                "equalize direction w_B - mu_B is zero"
            )
        equalized[row] = nu + scale * direction / norm
    return equalized


def hard_debias(
    E: np.ndarray,
    g: np.ndarray,
    neutral_words: Sequence[int],
    equality_sets: Sequence[Sequence[int]],
) -> np.ndarray:
    """
    Bolukbasi's hard debiasing: neutralize the neutral words, equalize the sets.

    Copies ``E`` and applies :func:`neutralize` to every gender-neutral word
    (the occupations -- their direct bias on ``g`` goes to zero) and
    :func:`equalize` to every equality set (the gendered pairs that must stay
    distinct but symmetric). The definitional words that defined ``g`` are left
    untouched, as in Bolukbasi: neutralizing ``he = +e0`` would annihilate it.

    Parameters
    ----------
    E : np.ndarray
        The embedding matrix, shape ``(V, d)``, unit-norm rows.
    g : np.ndarray
        The unit gender direction, shape ``(d,)``.
    neutral_words : Sequence[int]
        Row indices of the gender-neutral words to neutralize.
    equality_sets : Sequence[Sequence[int]]
        Row indices of each equality set to equalize, disjoint from
        ``neutral_words``.

    Returns
    -------
    np.ndarray
        The debiased matrix, shape ``(V, d)`` -- a copy; ``E`` is not mutated.

    Raises
    ------
    ValueError
        If a word appears in both ``neutral_words`` and an equality set (the
        two operations disagree on such a word, and the result would depend on
        the order they run in), or if :func:`equalize` raises.
    """
    neutral = list(neutral_words)
    sets = [list(equality_set) for equality_set in equality_sets]

    overlap = sorted(set(neutral) & {i for equality_set in sets for i in equality_set})
    if overlap:
        raise ValueError(
            f"rows {overlap} are both neutralized and equalized; the two sets "
            "must be disjoint"
        )

    debiased = E.copy()
    for i in neutral:
        debiased[i] = neutralize(E[i], g)
    for equality_set in sets:
        debiased[equality_set] = equalize(E[equality_set], g)
    return debiased


def weat_association(w: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    """
    WEAT's association score ``s(w, A, B)`` for one word.

    How much more similar ``w`` is to attribute set ``A`` than to ``B``::

        s(w, A, B) = mean_{a in A} cos(w, a) - mean_{b in B} cos(w, b)

    Positive means "leans A", negative "leans B", zero "unassociated". Cosines
    come from the head's :func:`~dlhub.embeddings.word_embeddings.cosine_similarity`.

    Parameters
    ----------
    w : np.ndarray
        The word's vector, shape ``(d,)``.
    A : np.ndarray
        The first attribute set's vectors, shape ``(|A|, d)``, one word per row.
    B : np.ndarray
        The second attribute set's vectors, shape ``(|B|, d)``.

    Returns
    -------
    float
        The association score ``s(w, A, B)``.
    """
    mean_a = float(np.mean([cosine_similarity(w, a) for a in A]))
    mean_b = float(np.mean([cosine_similarity(w, b) for b in B]))
    return mean_a - mean_b


def weat_statistic(X: np.ndarray, Y: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    """
    WEAT's test statistic: the total association gap between the target sets.

    ::

        statistic = sum_{x in X} s(x, A, B) - sum_{y in Y} s(y, A, B)

    with ``s`` from :func:`weat_association`. Large positive means the ``X``
    targets lean toward ``A`` more than the ``Y`` targets do.

    Parameters
    ----------
    X : np.ndarray
        The first target set's vectors, shape ``(|X|, d)``.
    Y : np.ndarray
        The second target set's vectors, shape ``(|Y|, d)``.
    A : np.ndarray
        The first attribute set's vectors, shape ``(|A|, d)``.
    B : np.ndarray
        The second attribute set's vectors, shape ``(|B|, d)``.

    Returns
    -------
    float
        The test statistic.
    """
    total_x = sum(weat_association(x, A, B) for x in X)
    total_y = sum(weat_association(y, A, B) for y in Y)
    return float(total_x - total_y)


def weat_effect_size(
    X: np.ndarray, Y: np.ndarray, A: np.ndarray, B: np.ndarray
) -> float:
    """
    WEAT's effect size ``d``: the association gap in pooled standard deviations.

    ::

        d = (mean_{x in X} s(x) - mean_{y in Y} s(y)) / std_{w in X u Y} s(w)

    with ``s`` from :func:`weat_association`` and the **population** std
    (``ddof = 0``; see the module Notes). ``d`` is scale-free -- it says how
    far apart the two target groups sit relative to their own spread -- and it
    is the dossier's hard known answer, before and after debiasing.

    Parameters
    ----------
    X : np.ndarray
        The first target set's vectors, shape ``(|X|, d)``.
    Y : np.ndarray
        The second target set's vectors, shape ``(|Y|, d)``.
    A : np.ndarray
        The first attribute set's vectors, shape ``(|A|, d)``.
    B : np.ndarray
        The second attribute set's vectors, shape ``(|B|, d)``.

    Returns
    -------
    float
        The effect size ``d``, or ``0.0`` when the pooled std is ``0`` (the
        zero-spread guard -- exactly the post-debias case; see the module
        Notes).
    """
    s_x = [weat_association(x, A, B) for x in X]
    s_y = [weat_association(y, A, B) for y in Y]

    spread = float(np.array(s_x + s_y).std())  # population std (ddof = 0)
    if spread == 0.0:
        # Every score is identical: 0/0 is defined to be 0.0, not nan.
        return 0.0
    return float((np.mean(s_x) - np.mean(s_y)) / spread)


def weat_pvalue(X: np.ndarray, Y: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    """
    WEAT's one-sided p-value, by **exact** enumeration of every permutation.

    Pools the targets ``X u Y`` and enumerates all ``C(|X u Y|, |X|)`` ways to
    choose a new ``X'`` of the same size (the complement is ``Y'``), computing
    the statistic of each partition. The p-value is the fraction of partitions
    whose statistic is ``>=`` the observed one -- the probability that a random
    equal-size split associates the targets with the attributes at least as
    strongly as the true split does.

    Exact enumeration replaces Caliskan et al.'s 10,000 sampled permutations;
    it is affordable here because the toy sets give only ``C(6, 3) = 20``
    partitions (see the module Notes).

    Parameters
    ----------
    X : np.ndarray
        The first target set's vectors, shape ``(|X|, d)``.
    Y : np.ndarray
        The second target set's vectors, shape ``(|Y|, d)``.
    A : np.ndarray
        The first attribute set's vectors, shape ``(|A|, d)``.
    B : np.ndarray
        The second attribute set's vectors, shape ``(|B|, d)``.

    Returns
    -------
    float
        The permutation p-value in ``[1/C, 1]``.

    Raises
    ------
    ValueError
        If ``|X| != |Y|`` -- the permutation test redistributes the pooled
        targets into *equal-size* partitions.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"equal-size target sets required; got |X| = {X.shape[0]} and "
            f"|Y| = {Y.shape[0]}"
        )

    observed = weat_statistic(X, Y, A, B)
    pool = np.vstack([X, Y])
    n, k = pool.shape[0], X.shape[0]

    at_least = 0
    total = 0
    for chosen in itertools.combinations(range(n), k):
        chosen_set = set(chosen)
        X_permuted = pool[list(chosen)]
        Y_permuted = pool[[j for j in range(n) if j not in chosen_set]]
        if weat_statistic(X_permuted, Y_permuted, A, B) >= observed:
            at_least += 1
        total += 1
    return at_least / total


def neutral_neighbors(E: np.ndarray, neutral_words: Sequence[int]) -> dict[int, int]:
    """
    Map every neutral word to its nearest neighbour *within the neutral set*.

    The tested half of the surviving-bias metric: :func:`hard_debias` zeroes
    every direct projection, but if this map is identical before and after,
    the words still *group by gender* through the indirect cluster axis -- the
    Gonen & Goldberg climax. Ranking reuses the head's
    :func:`~dlhub.embeddings.word_embeddings.nearest_neighbors` (cosine, self
    excluded, ties by ascending index), on the neutral sub-matrix alone.

    Parameters
    ----------
    E : np.ndarray
        The embedding matrix, shape ``(V, d)``, one word per row.
    neutral_words : Sequence[int]
        Row indices of the gender-neutral words, at least two.

    Returns
    -------
    dict[int, int]
        ``{row: neighbour_row}`` -- each neutral word's row index in ``E``
        mapped to its nearest neutral neighbour's row index in ``E``.

    Raises
    ------
    ValueError
        If fewer than two neutral words are given (there would be no neighbour
        to find).
    """
    words = list(neutral_words)
    if len(words) < 2:
        raise ValueError(f"need at least 2 neutral words; got {len(words)}")

    sub = E[words]  # (n, d): the neutral words only, in the given order
    mapping: dict[int, int] = {}
    for position, row in enumerate(words):
        # The head works on (d, V) columns: pass the sub-matrix transposed, so
        # its column j is neutral word j, then map the answer back to E's rows.
        neighbour, _score = nearest_neighbors(sub.T, position, k=1)[0]
        mapping[row] = words[neighbour]
    return mapping


def main() -> None:
    """Run the debiasing pipeline end to end and print its known answers."""
    fixture = make_fixture()
    E, vocab, ix = fixture["E"], fixture["vocab"], fixture["word_to_ix"]

    pairs = [(ix[male], ix[female]) for male, female in fixture["definitional_pairs"]]
    g = gender_direction(E, pairs)
    print(f"gender direction g = {np.round(g, 4)} (male-positive)")

    print("direct bias (projection on g):")
    for word in fixture["neutral_words"]:
        print(f"  {word:12s} {E[ix[word]] @ g:+.5f}")

    sets = fixture["weat_sets"]
    A, B = _rows(E, ix, sets["A"]), _rows(E, ix, sets["B"])
    X, Y = _rows(E, ix, sets["X"]), _rows(E, ix, sets["Y"])
    print(
        f"WEAT before: statistic = {weat_statistic(X, Y, A, B):.10f}, "
        f"d = {weat_effect_size(X, Y, A, B):.10f}, p = {weat_pvalue(X, Y, A, B)}"
    )

    neutral = [ix[word] for word in fixture["neutral_words"]]
    equality_sets = [[ix[word] for word in s] for s in fixture["equality_sets"]]
    debiased = hard_debias(E, g, neutral, equality_sets)

    X_after, Y_after = _rows(debiased, ix, sets["X"]), _rows(debiased, ix, sets["Y"])
    print(
        f"WEAT after:  statistic = {weat_statistic(X_after, Y_after, A, B):.10f}, "
        f"d = {weat_effect_size(X_after, Y_after, A, B):.10f}, "
        f"p = {weat_pvalue(X_after, Y_after, A, B)}"
    )

    before = neutral_neighbors(E, neutral)
    after = neutral_neighbors(debiased, neutral)
    print(f"neutral-neighbour map unchanged by debiasing: {before == after}")
    for row in neutral:
        print(f"  {vocab[row]:12s} -> {vocab[after[row]]}")

    for equality_set in equality_sets:
        members = ", ".join(vocab[i] for i in equality_set)
        print(f"equalize {{{members}}}:")
        for i in equality_set:
            print(f"  {vocab[i]:12s} {np.round(debiased[i], 5)}")


if __name__ == "__main__":
    main()
