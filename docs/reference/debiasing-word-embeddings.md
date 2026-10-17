# Debiasing word embeddings

This page is the lookup reference for **bias geometry** in a given static
embedding: how gender bias lives in a *direction* of the vector space, how
Bolukbasi et al.'s hard debiasing removes it (identify the subspace,
neutralize, equalize), and how WEAT (Caliskan, Bryson & Narayanan, 2017)
measures the association that remains — used here to quantify bias *before*
and *after* debiasing. It states the equations and rules and makes no
argument for them; the reasoning behind the mechanics catalogued here — why
bias is a direction, why removing it is "Lipstick on a Pig", and where
geometric methods give way to learning-time ones — is in this topic's
explanation view, reachable from the [Crosswalk](#crosswalk).

Notation departs from the head [word embeddings](word-embeddings.md) page
deliberately: this topic stores the embedding matrix in the $(V \times d)$
**row** convention — a word's vector is `E[i]`, the transpose of the head's
column layout — because projections, WEAT, and neighbour structure are all
transpose-free on rows. What a word vector *is* and how it is *learned* live
on the head and the [word2vec-and-glove](word2vec-and-glove.md) pages;
this topic treats the vectors as **given** — a small, hand-planted toy
matrix (see [Fixture and tolerances](#fixture-and-tolerances)) whose every
answer is exact and checkable by eye. Cosine similarity and nearest-neighbour
ranking are imported from the head rather than restated; this module studies
the bias geometry on top of them.

## Contents

- [Notation and shapes](#notation-and-shapes)
- [The $(V \times d)$ layout and the head relationship](#the-v-times-d-layout-and-the-head-relationship)
- [The gender direction](#the-gender-direction)
- [Neutralize](#neutralize)
- [Equalize](#equalize)
- [Hard debiasing: the pipeline](#hard-debiasing-the-pipeline)
- [WEAT](#weat)
- [The surviving-bias metric](#the-surviving-bias-metric)
- [Public surface](#public-surface)
- [Fixture and tolerances](#fixture-and-tolerances)
- [Failure modes](#failure-modes)
- [Implementation Examples](#implementation-examples)
- [Key Takeaways](#key-takeaways)
- [References](#references)
- [Crosswalk](#crosswalk)

## Notation and shapes

| Symbol | Meaning |
| --- | --- |
| $V$ | vocabulary size: the number of distinct words ($V = 14$ in the fixture) |
| $d$ | embedding dimension: the number of features per word ($d = 4$ in the fixture) |
| $E$ | the embedding matrix, shape $(V \times d)$ — one word per **row** (this topic's convention) |
| $i$ | a word's integer index, $0 \le i < V$ |
| $e_w$ | the embedding of word $w$: a $(d,)$ vector, the row $E[i]$ |
| $B$ | a *generalized* gender **subspace** (Bolukbasi §5); a single direction `g` is the rank-1 case ($k = 1$) |
| `vocab` | the word list, **sorted**, which fixes the row order of $E$ and the `word_to_ix` index map |

Every row of $E$ is **unit-norm** (`make_fixture` L2-normalizes; see
[Fixture and tolerances](#fixture-and-tolerances)). Unit-norm is required by
[equalize](#equalize) (its $\sqrt{1 - \lVert \nu \rVert^2}$ term is real only
on the unit sphere) and it keeps cosines comparable across the operations.
The matrix stores `float64`.

## The $(V \times d)$ layout and the head relationship

The head writes the embedding matrix as $E_{\text{head}}$ of shape $(d
\times V)$ — one word per **column** — and a word's vector as the column
$E_{\text{head}, :\,i}$. To stay transpose-free on projections, WEAT, and
neighbour ranking, this topic takes the transpose and works row-wise:

$$E \;=\; E_{\text{head}}^{\top} \;\in\; \mathbb{R}^{V \times d}, \qquad e_w \;=\; E[i] \;\in\; \mathbb{R}^d.$$

| This page | Head (`word-embeddings.md`) |
| --- | --- |
| $E$, shape $(V \times d)$ | $E_{\text{head}}$, shape $(d \times V)$, with $E = E_{\text{head}}^{\top}$ |
| $e_w = E[i]$ (a row) | $e_w = E_{\text{head}, :\,i}$ (a column) |
| `E[i]` | `E_head[:, i]` |

Where the head's column-convention helpers are reused (cosine similarity and
nearest neighbours), the row matrix is passed transposed (its columns are
then the rows of this page's $E$). The dossier settles this convention
because every operation in this topic — projection on $g$, the neutralize
formula, the WEAT `s` and its pooled standard deviation, and the neighbour
ranking — is a row-level computation.

## The gender direction

Given _definitional pairs_ `(he, she)`, `(man, woman)`, `(boy, girl)`, the
gender direction `g` is the **top principal component** ($k = 1$) of the
centered pair differences (Bolukbasi et al., 2016, §5.1). The procedure:

1. For each pair $(w_{\text{male}}, w_{\text{female}})$, take the pair mean
   $\mu = (e_{\text{male}} + e_{\text{female}}) / 2$ and the two deviations
   $e_{\text{male}} - \mu$ and $e_{\text{female}} - \mu$.
2. Stack every deviation into one deviation matrix (shape $(2n_{\text{pairs}},
   d)$).
3. Take the SVD; the **first right singular vector** is the unit gender
   direction $g$.

The SVD fixes each principal direction only up to sign, so $g$ is pinned
**male-positive** by convention: flip $g$ when $g \cdot (e_{\text{he}} -
e_{\text{she}}) < 0$ for the first definitional pair — the same
determinism-by-convention the head's `pca_project_2d` uses. Without the sign
rule, two runs could report mirrored projections and a WEAT statistic with
a flipped sign.

The method generalizes to a $k$-dimensional subspace $B$ by taking the top
$k$ right singular vectors instead of one; the projection onto $B$ becomes
$v_B = \sum_{j=1}^{k} (v \cdot b_j)\,b_j$. The fixture's deviation matrix is
exactly rank-1 — singular values $[\sqrt{6}, 0, 0, 0]$ — so one direction
*is* the whole subspace here, and every function takes a single unit
vector $g$.

`gender_direction(E, definitional_pairs)` raises `ValueError` when
`definitional_pairs` is empty.

## Neutralize

For a gender-neutral word $w$, **remove the $g$ component and renormalize**:

$$w' \;=\; \frac{w \;-\; (w \cdot g)\,g}{\bigl\lVert w \;-\; (w \cdot g)\,g \bigr\rVert}.$$

After this, $w' \cdot g = 0$ (the direct projection on the gender axis is
zero) and $w'$ is back on the unit sphere, so cosines and distances stay
comparable.

**Zero-residual guard.** A word that lies entirely along $g$ has no neutral
part: $w - (w \cdot g)\,g$ is the zero vector and renormalizing would
divide by zero. `neutralize` returns the zero vector in that case, which
the head's cosine zero-guard then keeps out of every ranking. The fixture's
definitional words (`he`, `she`, …) are exactly such pure-gender vectors,
and `hard_debias` is structured never to call `neutralize` on them — those
words *defined* $g$ and are left untouched, as in Bolukbasi.

## Equalize

For an equality set $E$ (a gendered pair to keep distinct, e.g.
`{grandfather, grandmother}`), Bolukbasi's Step 2a (arXiv:1607.06520)
replaces every member's neutral part with the set's shared neutral part
$\nu$ and gives the members equal-and-opposite $g$ components. With

$$\mu \;=\; \text{mean}(E), \qquad \mu_B \;=\; (\mu \cdot g)\,g, \qquad \nu \;=\; \mu - \mu_B, \qquad w_B \;=\; (w \cdot g)\,g,$$

each member $w \in E$ becomes

$$w' \;=\; \nu \;+\; \sqrt{1 - \lVert \nu \rVert^2} \;\cdot\; \frac{w_B - \mu_B}{\lVert w_B - \mu_B \rVert}.$$

Every output is unit-norm, the members' $g$ projections are $\pm$ the
shared $\sqrt{1 - \lVert \nu \rVert^2}$, and every member is **equidistant
from any neutralized word** — which is what makes the pair unbiased without
merging the two words into one vector.

| Property | Value |
| --- | --- |
| Inputs | the set's member rows, all **unit-norm** (normalize-first) |
| Outputs | unit-norm; shared neutral part $\nu$; equal-and-opposite $g$ components |
| Required condition | $\lVert \nu \rVert \le 1$ (otherwise $\sqrt{1 - \lVert \nu \rVert^2}$ is not real) |
| Failure mode | `ValueError` when $\lVert \nu \rVert > 1$ (non-unit inputs) or when a member's $g$ component equals the set mean's and the direction $w_B - \mu_B$ is zero |

## Hard debiasing: the pipeline

`hard_debias(E, g, neutral_words, equality_sets)` is the one-line pipeline
that applies the two operations above:

1. Copy $E$.
2. For every row index in `neutral_words` (the gender-neutral occupations),
   set $E'[i] = \text{neutralize}(E'[i], g)$. The direct bias — the
   projection $E'[i] \cdot g$ — becomes exactly $0$.
3. For every equality set in `equality_sets`, set $E'[\text{set}] =
   \text{equalize}(E[\text{set}], g)$. The set's $g$ components become
   $\pm \sqrt{1 - \lVert \nu \rVert^2}$ symmetrically about $g$.

The definitional words that defined $g$ are left untouched, as in Bolukbasi:
neutralizing `he = +e_0` would annihilate it. The two role lists must be
**disjoint**; `hard_debias` raises `ValueError` on overlap rather than
return a result whose order-of-operations would silently decide.

## WEAT

The Word Embedding Association Test (Caliskan, Bryson & Narayanan, 2017)
measures the differential association between two **attribute** sets $A$,
$B$ and two **target** sets $X$, $Y$. For an attribute set this topic uses
the male / female definitional words; for a target set, male-coded vs
female-coded occupations.

### Association score

For a single word $w$ and attribute sets $A$, $B$:

$$s(w, A, B) \;=\; \text{mean}_{a \in A}\,\cos(w,\,a) \;-\; \text{mean}_{b \in B}\,\cos(w,\,b).$$

Positive means "leans $A$", negative "leans $B$", zero "unassociated".
Cosines come from the head's `cosine_similarity` (zero-vector guard
included).

### Test statistic

The total association gap between the target sets:

$$\text{statistic} \;=\; \sum_{x \in X} s(x) \;-\; \sum_{y \in Y} s(y).$$

Large positive means the $X$ targets lean toward $A$ more than the $Y$
targets do.

### Effect size `d`

The gap expressed in pooled standard deviations of the per-target
association scores:

$$d \;=\; \frac{\text{mean}_{x \in X} s(x) \;-\; \text{mean}_{y \in Y} s(y)}{\text{std}_{w \in X \cup Y}\, s(w)}.$$

The standard deviation is the **population** standard deviation
(`ddof = 0`) of the pooled $s$-values; under `ddof = 1` the fixture's
validated answer would be $1.8102397229$ instead of $1.9830182615$.
$d$ is scale-free: it says how far apart the two target groups sit
relative to their own spread.

**Zero-spread guard.** When every pooled $s$-value is equal (no weighted
scatter), the std is zero and $d = 0/0$; `weat_effect_size` returns $0.0$ by
definition rather than a `nan`. This is **exactly the post-debias case**:
every neutralized word scores $0$ against the pure-gender attributes, so
the std is $0$, and the dossier pins $d_{\text{after}} = 0$ to this guard.

### Exact permutation `p`

`weat_pvalue(X, Y, A, B)` pools the targets $X \cup Y$ and enumerates
**every** $\binom{|X \cup Y|}{|X|}$ equal-size partition $(X', Y')$,
computing the statistic of each. The one-sided p-value is the fraction of
partitions whose statistic is $\ge$ the observed one — the probability
that a random equal-size split associates the targets with the attributes
at least as strongly as the true split does. The observed partition
always counts itself, so $p \ge 1 / \binom{n}{k}$.

`weat_pvalue` raises `ValueError` when $|X| \ne |Y|$ (the test redistributes
the pooled targets into *equal-size* partitions). Exact enumeration
replaces Caliskan et al.'s 10,000 sampled permutations; it is affordable
here because the fixture's $3$-vs-$3$ target sets give only
$\binom{6}{3} = 20$ partitions.

## The surviving-bias metric

Tested half: for each gender-neutral word, **its nearest neighbour within
the neutral set** — cosine, self excluded, ties broken by ascending index
(reusing the head's `nearest_neighbors` on the neutral sub-matrix alone).
The map is `{row: neighbour_row}`. If `horizontal neighbours` is identical
*before and after* `hard_debias`, the words still **group by gender** through
the indirect cluster axis the debiasing never touched — the
Gonen–Goldberg climax (Gonen & Goldberg, 2019).

Shown half (the notebook, not this page): a 2-D PCA scatter of the
debiased words coloured by original gender (the head's `pca_project_2d`),
in which the two clusters persist though every direct projection is $0$.

## Public surface

All names are re-exported from `dlhub.embeddings` (NumPy is always present,
so the re-export is eager; `make_fixture` is reached as
`dlhub.embeddings.debiasing_word_embeddings.make_fixture` so it does not
collide with the head's). Full docstrings are on the generated
[API page](api/embeddings.md); this is the at-a-glance map. Parameters and
return shapes match the implementation exactly.

| Function | Returns |
| --- | --- |
| `make_fixture()` | the planted bundle `{"E", "vocab", "word_to_ix", "dims", "definitional_pairs", "neutral_words", "equality_sets", "weat_sets"}` |
| `gender_direction(E, definitional_pairs)` | the unit gender direction $g$, shape $(d,)$, male-positive; raises `ValueError` on empty `definitional_pairs` |
| `neutralize(w, g)` | the neutralized unit vector, shape $(d,)$; the zero vector when $w$ lies entirely along $g$ |
| `equalize(vectors, g)` | the equalized vectors, shape $(n, d)$, in the input's member order; raises `ValueError` when $\lVert \nu \rVert > 1$ or a member's $g$ component equals the set mean's |
| `hard_debias(E, g, neutral_words, equality_sets)` | the debiased matrix, shape $(V, d)$ — a copy; raises `ValueError` when the two role lists overlap |
| `weat_association(w, A, B)` | the association score $s(w, A, B)$ (`float`) |
| `weat_statistic(X, Y, A, B)` | the WEAT test statistic (`float`) |
| `weat_effect_size(X, Y, A, B)` | the effect size $d$ (`float`); `0.0` when the pooled std is $0$ |
| `weat_pvalue(X, Y, A, B)` | the exact permutation p-value in $[1 / \binom{n}{k}, 1]$; raises `ValueError` on unequal target sizes |
| `neutral_neighbors(E, neutral_words)` | `dict[int, int]` mapping each neutral word's row to its nearest neutral neighbour's row; raises `ValueError` on fewer than two neutral words |

`cosine_similarity` and `nearest_neighbors` are imported from
`dlhub.embeddings.word_embeddings`, not reimplemented.

## Fixture and tolerances

The implementation and its tests share one **planted** toy embedding set,
`M = make_fixture()`. It is committed as an integer literal, never
downloaded or trained, so it cannot drift; every answer is exact and
human-checkable by reading the numbers.

- **Shape:** $V = 14$ words by $d = 4$, one word per row (= the head's
  $E_{\text{head}}^{\top}$). dtype `float64`.
- **Coordinates:** coordinate 0 is *gender* (the bias axis — `+1`
  masculine, $-1$ feminine); coordinate 1 is *occupation cluster* (the
  *indirect*-bias axis, correlated with gender — this is what survives
  neutralize); coordinates 2–3 are *within-cluster content* (tie-free
  neighbours).
- **Vocabulary** (sorted, fixing row order and `word_to_ix`):
  `boy, engineer, girl, grandfather, grandmother, he, man, mechanic, nanny,
  nurse, programmer, receptionist, she, woman`.

Raw rows before `make_fixture`'s row-wise L2 normalization:

| Word(s) | Role | Raw $(\text{gender},\ \text{cluster},\ \text{within}_a,\ \text{within}_b)$ |
| --- | --- | --- |
| `he`, `man`, `boy` | definitional (male) | $(1, 0, 0, 0)$ |
| `she`, `woman`, `girl` | definitional (female) | $(-1, 0, 0, 0)$ |
| `engineer` | male-coded occupation | $(1, 3, 3, 0)$ |
| `programmer` | male-coded occupation | $(1, 3, 2, 0)$ |
| `mechanic` | male-coded occupation | $(1, 3, 0, 0)$ |
| `nurse` | female-coded occupation | $(-1, -3, 3, 0)$ |
| `receptionist` | female-coded occupation | $(-1, -3, 2, 0)$ |
| `nanny` | female-coded occupation | $(-1, -3, 0, 0)$ |
| `grandfather` | equality set | $(2, 0, 0, 1)$ |
| `grandmother` | equality set | $(-1, 0, 0, 1)$ |

**The definitional six are pure-gender archetypes and coincide within
gender** (`he = man = boy = +e_0`, `she = woman = girl = -e_0`). This is
intentional: their only role is to define $g$ and to serve as WEAT's
gender attributes, and nothing computes neighbours among them — so the
coincidence keeps every known answer exact without affecting any result.

**WEAT sets.** Attributes $A = \{\text{He},\ \text{man},\ \text{boy}\}$
(male), $B = \{\text{she},\ \text{woman},\ \text{girl}\}$ (female); targets
$X = \{\text{engineer},\ \text{programmer},\ \text{mechanic}\}$ (male-coded),
$Y = \{\text{nurse},\ \text{receptionist},\ \text{nanny}\}$ (female-coded).
$|X| = |Y| = 3$, so the exact permutation test enumerates
$\binom{6}{3} = 20$ equal-size partitions.

**Known answers (the correctness anchors, validated numerically 2026-10-14):**

- **Gender direction.** $g = +e_0$ exactly; the SVD of the
  definitional-pair deviations has singular values $[\sqrt{6}, 0, 0, 0]$
  (rank-1), so a single direction ($k = 1$) captures the subspace.
- **Direct bias** (projection on $g$): $\text{engineer},\ \text{programmer},
  \ \text{mechanic} = 1/\sqrt{19},\ 1/\sqrt{14},\ 1/\sqrt{10} \approx
  0.22942,\ 0.26726,\ 0.31623$; the female occupations are the exact
  negatives.
- **WEAT before debiasing:** effect size $d = 1.9830182615$, statistic
  $3.2516189672$, exact permutation $p = 1/20 = 0.05$ (the true split
  uniquely maximizes the statistic).
- **After `hard_debias`:** every projection on $g$ is $0$, $d = 0$ exactly
  (the zero-spread guard), and $p = 1.0$. Equalize acts only on the
  equality set and does not change the WEAT over $X$ / $Y$.
- **Survival (the Gonen–Goldberg climax):** the neutral-neighbour map is
  **identical before and after** neutralize — `engineer ↔ programmer`,
  `mechanic → programmer`, `nurse ↔ receptionist`, `nanny → receptionist`.
  The gendered grouping is intact though every direct projection is $0$.
- **Equalize** `{grandfather, grandmother}`: the asymmetric inputs
  $(2, 0, 0, 1)/\sqrt{5}$ and $(-1, 0, 0, 1)/\sqrt{2}$ become
  $(\pm 0.81663, 0, 0, 0.57716)$ — unit-norm, symmetric about $g$, and
  equidistant ($\sqrt{2}$) from any neutralized word.

**Tolerances:**

| Comparison | Tolerance |
| --- | --- |
| $g$, projections, $d_{\text{after}}$, $p$, the neighbour map, equalize symmetry and norms | exact or `atol = 1e-12` |
| $d_{\text{before}}$ and $\text{stat}_{\text{before}}$ | `atol = 1e-9` |
| Equalize output vectors | `atol = 1e-9` |

## Failure modes

| Symptom | Cause | Fix |
| --- | --- | --- |
| Two runs report mirrored $g$ and a WEAT statistic with a flipped sign | the SVD's per-direction sign was left unfixed | flip $g$ when $g \cdot (e_{\text{he}} - e_{\text{she}}) < 0$ for the first definitional pair |
| `equalize` raises `ValueError` `||nu|| > 1` | the equality set's rows are not unit-norm | L2-normalize every row first (the fixture does this in `make_fixture`); equalize's $\sqrt{1 - \lVert \nu \rVert^2}$ is real only on the unit sphere |
| `equalize` raises `ValueError` `direction undefined` | a member's $g$ component equals the set mean's, so $w_B - \mu_B = 0$ | remove that member from the equality set, or split the set |
| `neutralize` returns the zero vector | $w$ lies entirely along $g$ (pure-gender word, no neutral part) | expected for definitional words; do **not** neutralize them — `hard_debias` only neutralizes the gender-neutral role rows |
| `weat_effect_size` returns `nan` | the pooled std is zero but no guard was in place | the guard returns $0.0$ by definition; this is the post-debias case, not a bug |
| $d_{\text{before}}$ matches $1.81\ldots$ instead of $1.983\ldots$ | the sample std (`ddof = 1`) was used | divide by the **population** std (`ddof = 0`) — the fixture's known answer is pinned to this |
| `weat_pvalue` raises `ValueError` `equal-size target sets required` | the target sets have different sizes | the permutation test redistributes the pooled targets into equal-size partitions; use balanced target sets |
| After `hard_debias` the WEAT `d` is still large | `hard_debias` was not applied to the neutral words, or $g$ was computed from the wrong definitional pairs | pass every neutral word index and the corrected $g$; the WEAT sets are unchanged |
| Direct bias is $0$ yet the neutral-neighbour map is unchanged | the **indirect** cluster axis survived the debiasing — Gonen–Goldberg | this is the topic's thesis, not a bug; the notebook visualizes the surviving clusters |
| An equality-set member came back one vector | `hard_debias` and a neutralizer both targeted it | raise on overlap: the two operations would disagree on such a word, and the order they ran in would silently decide the result |
| `$g$` is a $k > 1$ subspace but every function takes one vector | a $k$-dimensional subspace needs a basis, not one direction | take the top $k$ right singular vectors; generalize the projection to $v_B = \sum_j (v \cdot b_j)\, b_j$ |

## Implementation Examples

### Complete Implementations:

> #### **[Debiasing word embeddings — from scratch (NumPy)](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/embeddings/debiasing_word_embeddings.py)** - The bias-direction geometry on top of the head's cosine and nearest-neighbour helpers: the planted $(V \times d)$ fixture with its role labels, the top-PC gender direction with its deterministic male-positive sign rule, the neutralize formula (with the pure-gender zero-residual guard), Bolukbasi's equalize formula exactly as arXiv:1607.06520 §A writes it (with its $\lVert \nu \rVert > 1$ and zero-direction guards), the `hard_debias` pipeline with its disjoint-role invariant, WEAT's `s` / statistic / effect size `d` (population std, `ddof = 0`, with the zero-spread guard) and the exact enumerated permutation `p`, and the `neutral_neighbors` map that proves the indirect bias survives. Checked against the fixture: $g = +e_0$ with the rank-1 singular spectrum, the exact projections, WEAT `d` / statistic / `p` before ($1.9830182615$ / $3.2516189672$ / $0.05$) and after-debias projections `0`, $d = 0$, $p = 1.0$, the neutral-neighbour map identical before and after, equalize symmetry / unit-norm / equidistance, and the `d` zero-spread guard. Exposes `make_fixture`, `gender_direction`, `neutralize`, `equalize`, `hard_debias`, `weat_association`, `weat_statistic`, `weat_effect_size`, `weat_pvalue`, and `neutral_neighbors`; imports `cosine_similarity` and `nearest_neighbors` from the head.

## Key Takeaways

1. The embedding matrix here is $E \in \mathbb{R}^{V \times d}$, one word per row, $E = E_{\text{head}}^{\top}$ — the transpose of the head's $(d \times V)$ column convention; a word's vector is $E[i]$, and every row is unit-norm.
2. The gender direction $g$ is the **top principal component** ($k = 1$) of the centered definitional-pair differences, taken via SVD of the stacked deviations, with a deterministic **male-positive** sign rule so $g \cdot (e_{\text{he}} - e_{\text{she}}) > 0$.
3. **Neutralize** is $w' = (w - (w \cdot g)\,g) / \lVert w - (w \cdot g)\,g \rVert$: project the $g$ component out and renormalize; a pure-gender $w$ returns the zero vector (the zero-residual guard) so it never poisons a ranking.
4. **Equalize** gives an equality set's members the shared neutral part $\nu$ and equal-and-opposite $g$ components: $w' = \nu + \sqrt{1 - \lVert \nu \rVert^2} \cdot (w_B - \mu_B) / \lVert w_B - \mu_B \rVert$, with $w_B = (w \cdot g)\,g$ — every output unit-norm, all equidistant from any neutralized word (Bolukbasi §A, arXiv:1607.06520).
5. **WEAT** reports an association $s(w, A, B) = \text{mean}_A \cos(w, a) - \text{mean}_B \cos(w, b)$; a statistic summing $s$ over $X$ and $Y$; an effect size $d$ in the **population** std (`ddof = 0`); and an exact permutation `p` that enumerates every $\binom{|X \cup Y|}{|X|}$ equal-size partition.
6. The **zero-spread guard** returns $d = 0.0$ (not `nan`) when the pooled std is zero — exactly the post-debias case, where every neutralized word scores $0$ against the pure-gender attributes.
7. After `hard_debias` the direct bias vanishes and WEAT says "fixed"; **the neutral-neighbour map is unchanged** (self excluded, ties by ascending index, within the neutral set) — the surviving indirect cluster axis is the Gonen–Goldberg "Lipstick on a Pig" critique in one map.
8. One planted `float64` fixture ($V = 14$, $d = 4$) fixes every answer exactly: $g = +e_0$, WEAT $d_{\text{before}} = 1.9830182615$ / statistic $3.2516189672$ / $p = 0.05$; $d_{\text{after}} = 0$ and $p = 1.0$; the equalize closed form $(\pm 0.81663, 0, 0, 0.57716)$; exact / `atol = 1e-12` for the exact anchors, `atol = 1e-9` for $d_{\text{before}}$ / $\text{stat}_{\text{before}}$ and the equalize output vectors.

## References

- **Bolukbasi, T., Chang, K.-W., Zou, J., Saligrama, V. & Kalai, A. (2016). Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings. NeurIPS.**  
  [https://arxiv.org/abs/1607.06520](https://arxiv.org/abs/1607.06520) – Hard debiasing: identify the gender subspace, neutralize the neutral words, equalize the equality sets. The exact equalize formula confirmed against §A before coding.
- **Caliskan, A., Bryson, J. J. & Narayanan, A. (2017). Semantics derived automatically from language corpora contain human-like biases. _Science_ 356(6334), 183–186.** – The Word Embedding Association Test: the association score `s(w, A, B)`, the test statistic, and the effect size `d`. This page uses the exact-enumeration `p` in place of their 10,000 sampled permutations.
- **Gonen, H. & Goldberg, Y. (2019). Lipstick on a Pig: Debiasing Methods Cover up Systematic Gender Biases. NAACL.**  
  [https://arxiv.org/abs/1903.03862](https://arxiv.org/abs/1903.03862) – Geometric debiasing is superficial: direct bias vanishes while neighbour structure (and downstream classifiers) still encode the bias. The thesis this topic demonstrates with the neutral-neighbour map.
- **Ng, A. (2018). Deep Learning Specialization, Course 5 (Sequence Models), Week 2 — Natural Language Processing & Word Embeddings.**  
  [https://www.coursera.org/specializations/deep-learning](https://www.coursera.org/specializations/deep-learning) – The family's framing; this topic treats the vectors it produces as given and studies the bias geometry on top of them.

## Crosswalk

| Reader wants to | Go to |
| --- | --- |
| Learn it step by step | [Debiasing word embeddings (notebook)](../tutorials/06-debiasing-word-embeddings.ipynb) |
| Look up an equation or rule | Debiasing word embeddings — *this page* |
| Understand why it works | [Debiasing word embeddings](../explanation/debiasing-word-embeddings.md) |
| Learn by running it | [Debiasing word embeddings (notebook)](../tutorials/06-debiasing-word-embeddings.ipynb) |