# Debiasing word embeddings

A word embedding is a *compressed summary* of the contexts its training words
appeared in. When those contexts include the social regularities of who does
what — which professions are coded masculine, which feminine — the geometry
carries those regularities in the same directions that carry every other
meaning. This page argues what that absorption does to the space: that it
takes the form of a *direction*, that the standard fix identifies that direction
and projects it away, and — as of 2019, when Gonen & Goldberg published their
critique — that the fix is cosmetic. The metric the fix is judged on reports
zero; the structure the metric does not measure reports otherwise.

It treats the vectors as *given* — a small, hand-planted toy matrix whose
every answer is exact and checkable by eye. What a word vector *is* and what
its geometry buys — cosine similarity, nearest neighbours, the parallelogram
analogy, and a 2-D PCA view — belong to the [word embeddings](word-embeddings.md)
topic; how the vectors were *learned* from a corpus belongs to the
[word2Vec-and-gloVe](word2vec-and-glove.md) topic; both are taken as read. For
the equations and exact rules in lookup form see this topic's
[reference](../reference/debiasing-word-embeddings.md); for the hands-on walk
see the notebook through the [Crosswalk](#crosswalk).

The notation matches the head. The matrix is laid out as $(V \times d)$ — one
word per *row*, the transpose of the head's column-major $E$ — and a word's
vector is $E[i] \in \mathbb{R}^d$. Every row is **unit-norm** (the fixture
L2-normalizes so equalize's $\sqrt{1 - \|\nu\|^2}$ stays real). A single
direction $g$ is taken — Bolukbasi et al.'s headline — which on this fixture
is the entire bias subspace, because the deviation matrix is exactly rank-1.

## Contents

- [Bias is a direction, not a coordinate](#bias-is-a-direction-not-a-coordinate)
- [From definitional pairs to a gender axis](#from-definitional-pairs-to-a-gender-axis)
- [Hard debiasing: neutralize and equalize](#hard-debiasing-neutralize-and-equalize)
- [WEAT measures the association that survives](#weat-measures-the-association-that-survives)
- [The metrics say "fixed"](#the-metrics-say-fixed)
- [The neighbourhood says otherwise](#the-neighbourhood-says-otherwise)
- [Why the indirect bias survives](#why-the-indirect-bias-survives)
- [Where geometric debiasing goes from here](#where-geometric-debiasing-goes-from-here)
- [Implementation Examples](#implementation-examples)
- [Key Takeaways](#key-takeaways)
- [References](#references)
- [Crosswalk](#crosswalk)

## Bias is a direction, not a coordinate

The head topic made one structural point this page relies on: the
*coordinates* of a learned embedding are not individually interpretable.
There is no axis you can point at and call "royalty" or "gender" — every
real embedding's axes are arbitrary rotations of every other, and the names a
corpus invites ("royalty", "gender") are not the arrows a thing would call
"the gender axis" in any pretrained model. What survives into the geometry
are the *directions and offsets* — the analogies, the neighbour clusters,
and, for this page, the bias. Bias lives as a direction.

Three things follow, and they shape the topic's structure. First, the right
question is never "is coordinate 37 positive?" but "is there a direction in
which the male and female definitional pairs differ?" — and that direction is
what the next section identifies. Second, removing the bias does not mean
deleting a column; it means *projecting out* the offending direction, which
on a unit-norm space is a normalisation step (the
[equalize](#hard-debiasing-neutralize-and-equalize) formula's
$\sqrt{1 - \|\nu\|^2}$ is the projection-onto-the-unit-sphere reminder).
Third, removing one direction leaves every other direction untouched — and
that, as the [last three sections](#the-neighbourhood-says-otherwise) argue,
is the geometric reason the fix is cosmetic. The metric "removing the bias"
naturally measures is the direct projection on the removed direction; the
metric the fix does *not* measure is everything the removed direction left
behind.

## From definitional pairs to a gender axis

Bolukbasi et al.'s identifying move (NeurIPS 2016) is to take a small list of
*definitional* pairs — `(he, she)`, `(man, woman)`, `(boy, girl)` — words
whose meanings differ only in gender, and to recover the gender direction
from how those pairs sit in the space. The procedure is the same principal-
component idea the head topic uses to flatten its line of sight, applied to
the centred *differences* of the pairs.

Concretely. For each pair $(w_{\text{male}}, w_{\text{female}})$ take the pair
mean $\mu = (e_{\text{male}} + e_{\text{female}}) / 2$ and the two deviations
$e_{\text{male}} - \mu$ and $e_{\text{female}} - \mu$. Stack every deviation
into one $(2 n_{\text{pairs}} \times d)$ matrix and take its SVD; the **first
right singular vector** is the unit gender direction $g$. The SVD fixes a
direction only up to sign, so $g$ is pinned **male-positive** by convention:
flip $g$ when $g \cdot (e_{\text{he}} - e_{\text{she}}) < 0$ for the first
definitional pair. Without the rule, two runs could report mirrored
projections and a WEAT statistic with a flipped sign.

The method generalises to a $k$-dimensional subspace $B$ by taking the top
$k$ right singular vectors instead of one; Bolukbasi et al. use $k = 10$ on
real pretrained vectors. On the topic's planted fixture the deviation matrix
is **exactly rank-1** — singular values $[\sqrt{6}, 0, 0, 0]$, so a single
direction is the entire subspace — and the procedure returns
$g = +e_0$ exactly. The definitional six (`he, man, boy` and `she, woman,
girl`) are pure-gender archetypes that coincide within gender (`he = man =
boy = +e_0`), which is why the answer is so clean: nothing in the deviation
matrix has a component off axis 0.

That clean answer is *teaching*, not realistic. The real task is to take a
pretrained embedding, identify the gender axis without being told its index,
and remove it — and the planted fixture exists to make every step of that
task exact and checkable by eye.

## Hard debiasing: neutralize and equalize

Once the gender direction $g$ is in hand, Bolukbasi et al.'s hard debiasing
does two things to a *copy* of the matrix: **neutralize** the gender-neutral
words, and **equalize** the equality sets.

**Neutralize.** For a gender-neutral word $w$ — an occupation, say, that
should not carry gender at all — remove the $g$ component and renormalise:

$$w' \;=\; \frac{w \;-\; (w \cdot g)\,g}{\bigl\lVert w \;-\; (w \cdot g)\,g \bigr\rVert}.$$

After this, $w' \cdot g = 0$ exactly: the direct projection on the gender
axis is gone. On the fixture the three male-coded occupations
(`engineer, programmer, mechanic`) start with projections $1/\sqrt{19},
1/\sqrt{14}, 1/\sqrt{10} \approx 0.229, 0.267, 0.316$ on $g = +e_0$; the
three female-coded occupations are their exact negatives. After
`hard_debias`, all six projections are exactly $0$. The guard against
pure-gender words (the residual $w - (w \cdot g) g$ is the zero vector when
$w$ lies entirely along $g$) returns the zero vector instead of dividing by
zero; the head's cosine zero-guard then keeps such a vector out of every
ranking.

**Equalize.** For an equality set $E$ — a gendered pair that has to *stay*
distinct, like `{grandfather, grandmother}` — Bolukbasi's Step 2a gives the
two members the **shared** neutral part of their mean and **equal-and-
opposite** gender components. With

$$\mu = \text{mean}(E),\quad \mu_B = (\mu \cdot g)\,g,\quad \nu = \mu - \mu_B,\quad w_B = (w \cdot g)\,g,$$

each member $w \in E$ becomes

$$w' \;=\; \nu \;+\; \sqrt{1 - \lVert \nu \rVert^2} \;\cdot\; \frac{w_B - \mu_B}{\lVert w_B - \mu_B \rVert}.$$

The output is unit-norm (the $\sqrt{1 - \|\nu\|^2}$ is real only on the
unit sphere — hence the "normalize first" rule), the two members' $g$
projections are $\pm$ the shared $\sqrt{1 - \|\nu\|^2}$, and every member
is **equidistant from any neutralized word**, which is what makes the
pair unbiased without merging the two words into one vector. On the
fixture the asymmetric inputs $(2, 0, 0, 1)/\sqrt{5}$ and
$(-1, 0, 0, 1)/\sqrt{2}$ become $(\pm\, 0.81663,\ 0,\ 0,\ 0.57716)$:
unit-norm, symmetric about $g$, and — because the fixture's neutral set
has no axis-3 content — exactly $\sqrt{2}$ from every neutralized word.
The reference page pins the formula against Bolukbasi §A (arXiv:1607.06520)
and lists the guards.

## WEAT measures the association that survives

The Word Embedding Association Test (Caliskan, Bryson & Narayanan, 2017)
asks how strongly two **target** sets of words are associated with two
**attribute** sets, using cosine as the currency. For a single word $w$ and
attribute sets $A$, $B$ the association is

$$s(w, A, B) \;=\; \text{mean}_{a \in A}\,\cos(w,\,a) \;-\; \text{mean}_{b \in B}\,\cos(w,\,b).$$

Positive means "leans $A$", negative "leans $B$", zero "unassociated". For
target sets $X$, $Y$ the **statistic** is the total gap,
$\sum_{x \in X} s(x) - \sum_{y \in Y} s(y)$; the **effect size** is the
gap expressed in pooled standard deviations,

$$d \;=\; \frac{\text{mean}_{x \in X}\, s(x) \;-\; \text{mean}_{y \in Y}\, s(y)}{\text{std}_{w \in X \cup Y}\, s(w)},$$

with the **population** standard deviation (`ddof = 0`) — under the sample
std the fixture's pinned $d = 1.9830\ldots$ would be $1.8102\ldots$.
Caliskan et al. sample 10,000 random partitions for the p-value; the
fixture's target sets are small enough (3-vs-3) to do an exact enumeration
over the $\binom{6}{3} = 20$ equal-size partitions, which is what this
topic reports. The p-value is the fraction of partitions whose statistic is
$\geq$ the observed one — the probability that a random equal-size split
associates the targets with the attributes at least as strongly as the
true split does.

On the fixture the attribute sets are the male / female definitional
words — $A = \{\text{he}, \text{man}, \text{boy}\}$,
$B = \{\text{she}, \text{woman}, \text{girl}\}$ — and the targets are the
male-coded / female-coded occupations — $X = \{\text{engineer},
\text{programmer}, \text{mechanic}\}$,
$Y = \{\text{nurse}, \text{receptionist}, \text{nanny}\}$. Before
debiasing the **statistic** is $3.2516\ldots$, the **effect size** is
$d = 1.9830\ldots$, and the **exact permutation p-value** is
$1/20 = 0.05$ — the true split uniquely maximises the statistic, so the
p-value is the floor.

## The metrics say "fixed"

Apply `hard_debias` and the metric the procedure is judged on says
"fixed". Every neutralised occupation has $w \cdot g = 0$ exactly, so every
$s(w, A, B)$ is $0$ (a vector orthogonal to all pure-gender attributes has
zero cosine to them, on either side), the pooled std is $0$, and `d` returns
$0.0$ by the zero-spread guard. With every partition's statistic equal to
$0$, all $\binom{6}{3} = 20$ partitions tie the observed one and $p = 1.0$.
The headline transitions are

| WEAT quantity | Before | After |
| --- | --- | --- |
| Statistic | $3.2516\ldots$ | $0$ |
| Effect size $d$ | $1.9830\ldots$ | $0$ (zero-spread guard) |
| Exact permutation $p$ | $1/20 = 0.05$ | $20/20 = 1.0$ |

and every direct projection on $g$ — engineer, programmer, mechanic, nurse,
receptionist, nanny — is $0$. The metric Bolukbasi et al. proposed to
measure the bias thinks the bias is gone.

## The neighbourhood says otherwise

This is the topic's thesis, demonstrated. After `hard_debias`, the *direct*
bias on the gender axis is exactly $0$ — but the **neutral-neighbour map**
(before and after, identical) shows the gender split has migrated
elsewhere. Run the head's `nearest_neighbors` on the neutral set alone —
`{engineer, programmer, mechanic, nurse, receptionist, nanny}` — with
cosine, self excluded, ties by ascending index, and the result is the
same map on the original matrix and on the debiased matrix:

| Word | Nearest neighbour in the neutral set |
| --- | --- |
| `engineer` | `programmer` |
| `programmer` | `engineer` |
| `mechanic` | `programmer` |
| `nurse` | `receptionist` |
| `receptionist` | `nurse` |
| `nanny` | `receptionist` |

The map has **two components**: a male-coded cluster (`engineer,
programmer, mechanic`) and a female-coded cluster (`nurse, receptionist,
nanny`). The two clusters point at each other; within each cluster the
words point at each other; nothing gendered has been moved by the
projection out of $g$. The notebook visualises the same point in 2-D with
the head's `pca_project_2d`: after debiasing, the debiased occupation
words coloured by original gender fall into two clusters — the same two
clusters that were there from the start.

That is the **Gonen & Goldberg** "Lipstick on a Pig" headline (NAACL 2019)
in one map: removing the projection on the gender axis does not remove the
neighbourhood structure that encodes gender. A downstream classifier that
reads cosine-to-the-nearest-neighbour would still see the two groups; a
downstream system that uses word embeddings as features for a profession-
from-pronoun task would still find the same correlations; the bias is
intact, only its *measure* has been retired. The metric the procedure was
designed to satisfy said the procedure worked; the structure the metric
does not see said otherwise.

## Why the indirect bias survives

The reason is geometric, and the planted fixture makes it exact. The
occupation words are not one-dimensional; each is a point on **two**
correlated axes, and the two axes carry the gender split *separately*.
On the fixture axis $0$ is the gender axis (the bias direction Bolukbasi
et al. identify as $g$) and axis $1$ is an *occupation cluster* axis — the
sign that puts engineers, programmers, mechanics on one side and nurses,
receptionists, nannies on the other. The raw occupation vectors split on
*both* axes: `engineer = (1, 3, 3, 0)`, `nurse = (-1, -3, 3, 0)`, and so
on. The two axes are correlated in the planted data (positive axis $0$ and
positive axis $1$ go together), but they are *different* axes — the
cluster axis carries the gender split independently.

`neutralize` projects out the $g$ component, which is axis $0$. It does
not touch axis $1$. After the projection, axis $0$ is reduced to $0$ and
the remaining vector on axes $1$–$3$ is renormalised to unit length — but
the sign of the axis-$1$ component is unchanged. So `engineer`'s
normalised post-neutralize vector still has a **positive** axis-$1$
component, and `nurse`'s still has a **negative** axis-$1$ component,
which is exactly what makes their nearest neighbours in the neutral set
be `programmer` and `receptionist` rather than each other.

In other words: the bias was *duplicated* across two axes. The
*direct* bias lived on axis $0$, the *indirect* bias lived on axis $1$.
Removing axis $0$ leaves axis $1$; the metrics that measure axis $0$
report success; the metrics that measure axis $1$ — or, equivalently,
the *neighbourhood structure* that axis $1$ encodes — report that the
bias is intact. The geometric procedure was sound; the premise that
bias lives in a single direction was not.

That is what Gonen & Goldberg (2019) report on real pretrained vectors:
the gender split is encoded in *many* directions, removing the top one
leaves the others, and downstream tasks — pronoun resolution, profession
classification, retrieval with hard-coded similarity — pick the surviving
ones back up. The critique is the same as on the fixture, and the fixture
makes it exact: a four-dimensional space small enough to read by eye is
the smallest case where "removing the bias" and "removing the structure
that encoded it" can be told apart, and on that space they are not the
same thing.

## Where geometric debiasing goes from here

Three threads continue from Bolukbasi et al. (2016); each addresses the
gap between *direct* and *indirect* bias a different way, and none of
them is implemented in this topic.

**Soft debiasing.** Bolukbasi et al. themselves propose a linear-transform
alternative: rather than project every word exactly onto $g^\perp$, learn
the linear map $E \mapsto E W$ that minimises gender projection while
preserving pairwise distances. The premise is the same (a gender direction
exists), the constraint is *soft*, and the answer is a continuous
trade-off. Hard debiasing is the exact-on-the-fixture headline; soft
debiasing is the operational form that scales.

**Learning-time debiasing.** The debiasing constraint can be baked into
the *training* of the embeddings, so the vectors are never biased in the
first place. Zhao et al. (2018) introduce a gender-neutral / gender-
specific split and a loss that pulls neutral words toward the midpoint of
their gendered counterparts; GN-GloVe modifies the GloVe objective
directly to penalise gender projection during co-occurrence fitting. The
geometry has no axis to project out at the end because the axis was
never built.

**Multiclass and racial bias.** Manzini et al. (2019) generalise the
identification-and-projection machinery to multiple demographic axes —
Black / white, Hispanic / non-Hispanic, Asian / non-Asian — and to
*intersectional* bias (e.g., Black women vs white men), and report both
the transferability of the approach and its limits. The fixture here is
gender-only, but the identification step (a direction) and the projection
step (a linear map) carry over.

For the equations, the failure modes, and the equalize formula pinned
against Bolukbasi §A, see the [reference](../reference/debiasing-word-embeddings.md);
for the hands-on walk — the recovered $g$, the WEAT before/after, the
neutral-neighbour map, and the 2-D PCA scatter that shows the surviving
clusters — see the notebook through the [Crosswalk](#crosswalk). The
*cautionary* reading is the topic's thesis: as of 2019, geometric
debiasing is widely understood to be cosmetic. Removing the direct
projection on the gender direction is a precise algebraic move; removing
the bias the geometry encodes is not.

## Implementation Examples

### Complete Implementations:

> #### **[Debiasing word embeddings — from scratch (NumPy)](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/embeddings/debiasing_word_embeddings.py)** - The bias-direction geometry on top of the head's cosine and nearest-neighbour helpers: the planted $(V \times d)$ fixture with its role labels, the top-PC gender direction with its deterministic male-positive sign rule, the neutralize formula (with the pure-gender zero-residual guard), Bolukbasi's equalize formula exactly as arXiv:1607.06520 §A writes it (with its $\lVert \nu \rVert > 1$ and zero-direction guards), the `hard_debias` pipeline with its disjoint-role invariant, WEAT's `s` / statistic / effect size `d` (population std, `ddof = 0`, with the zero-spread guard) and the exact enumerated permutation `p`, and the `neutral_neighbors` map that proves the indirect bias survives. Checked against the fixture: $g = +e_0$ with the rank-1 singular spectrum, the exact projections, WEAT `d` / statistic / `p` before ($1.9830182615$ / $3.2516189672$ / $0.05$) and after-debias projections `0`, $d = 0$, $p = 1.0$, the neutral-neighbour map identical before and after, equalize symmetry / unit-norm / equidistance, and the `d` zero-spread guard.

## Key Takeaways

1. Bias lives as a *direction* in the embedding space, not a coordinate —
   and once you accept the direction, "removing the bias" means projecting
   it out, not picking a column to drop.
2. Bolukbasi et al. identify the gender axis as the top principal
   component ($k = 1$) of the centred definitional-pair deviations, with
   a deterministic **male-positive** sign rule on the first pair.
3. **Neutralize** is $w' = (w - (w \cdot g)\, g) / \lVert w - (w \cdot g)\, g \rVert$:
   project the $g$ component out, renormalise. After it, every neutralised
   word's projection on $g$ is exactly $0$; the zero-residual guard
   protects pure-gender words from division by zero.
4. **Equalize** is $w' = \nu + \sqrt{1 - \lVert \nu \rVert^2} \cdot (w_B - \mu_B) / \lVert w_B - \mu_B \rVert$:
   shared neutral part plus equal-and-opposite $g$ components, all
   equidistant from any neutralised word. It needs unit-norm inputs, and
   the fixture's normalised rows give the closed form $(\pm\, 0.81663,
   0, 0, 0.57716)$.
5. **WEAT** measures association with $s(w, A, B) = \text{mean}_A \cos - \text{mean}_B \cos$,
   a target-set statistic, an effect size $d$ in the **population** std
   (`ddof = 0`), and an exact permutation p over $\binom{|X \cup Y|}{|X|}$
   equal-size partitions. The zero-spread guard returns $d = 0.0$ when
   the std is zero — exactly the post-debias case.
6. After `hard_debias` the *direct* bias vanishes and WEAT says "fixed"
   ($d: 1.983 \to 0$, $p: 0.05 \to 1.0$, every projection on $g$ zero);
   the *indirect* bias **survives** — the neutral-neighbour map is
   identical before and after (`engineer ↔ programmer`, `nurse ↔
   receptionist`, etc.).
7. The reason is geometric: on the planted fixture the occupation words
   sit on **two** correlated axes, gender and cluster. Removing axis $0$
   leaves axis $1$; the metrics that see axis $0$ report success; the
   metrics that see axis $1$ — equivalently, the neighbourhood structure
   — report that the bias is intact. The premise that bias lives in a
   single direction is the topic's claim to be wary of.
8. Where this goes from here: **soft debiasing** (Bolukbasi et al.'s
   linear-transform variant), **learning-time debiasing** (Zhao et al.
   2018, GN-GloVe), and **multiclass / racial bias** (Manzini et al.
   2019) — each named here, each deferred to the literature.

## References

- **Bolukbasi, T., Chang, K.-W., Zou, J., Saligrama, V. & Kalai, A. (2016). Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings. NeurIPS.**  
  [https://arxiv.org/abs/1607.06520](https://arxiv.org/abs/1607.06520) – Hard debiasing: identify the gender subspace, neutralize the neutral words, equalize the equality sets. The exact equalize formula is confirmed against §A before coding.
- **Caliskan, A., Bryson, J. J. & Narayanan, A. (2017). Semantics derived automatically from language corpora contain human-like biases. _Science_ 356(6334), 183–186.** – The Word Embedding Association Test: the association score $s(w, A, B)$, the test statistic, and the effect size $d$. This topic uses the exact-enumeration p in place of the paper's 10,000 sampled permutations.
- **Gonen, H. & Goldberg, Y. (2019). Lipstick on a Pig: Debiasing Methods Cover up Systematic Gender Biases in Word Embeddings. NAACL.**  
  [https://arxiv.org/abs/1903.03862](https://arxiv.org/abs/1903.03862) – Geometric debiasing is superficial: direct bias vanishes while the neighbourhood structure (and downstream classifiers) still encodes the bias. The thesis this topic demonstrates on the four-dimensional fixture.
- **Ng, A. (2018). Deep Learning Specialization, Course 5 (Sequence Models), Week 2 — Natural Language Processing & Word Embeddings.**  
  [https://www.coursera.org/specializations/deep-learning](https://www.coursera.org/specializations/deep-learning) – The source of this family's $(d \times V)$ $E$ notation, which this topic transposes.

## Crosswalk

| Reader wants to | Go to |
| --- | --- |
| Learn it step by step | [Debiasing word embeddings (notebook)](../tutorials/06-debiasing-word-embeddings.ipynb) |
| Look up an equation or rule | [Debiasing word embeddings](../reference/debiasing-word-embeddings.md) |
| Understand why it works | Debiasing word embeddings — *this page* |
| Learn by running it | [Debiasing word embeddings (notebook)](../tutorials/06-debiasing-word-embeddings.ipynb) |