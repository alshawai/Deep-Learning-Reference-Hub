# Word2Vec and GloVe

This page is the lookup reference for how static word embeddings are **learned**:
the two embedding tables and their shapes, the training-pair rule, the skip-gram,
CBOW, and GloVe objectives with their exact gradients, the smoothed negative
sampler, GloVe's co-occurrence weighting, the initialization and learning-rate
defaults, and the planted fixture with its tolerances. It states the equations
and rules and makes no argument for them.

Notation follows the sibling [word embeddings](word-embeddings.md)
reference but **departs** from it deliberately where training needs it (see
[Notation and shapes](#notation-and-shapes)). This topic *produces* the vectors
that page treats as given; the geometry read off them afterwards — cosine
similarity, nearest neighbors, the analogy rule — is that page's, and this one
links it rather than restating it. The reasoning behind the mechanics catalogued
here — why a predictive objective, why the $3/4$ exponent, why GloVe weights its
counts, and the Levy–Goldberg view that unifies the two families — is in this
topic's explanation view, reachable from the [Crosswalk](#crosswalk).

## Contents

- [Notation and shapes](#notation-and-shapes)
- [Training pairs and the window](#training-pairs-and-the-window)
- [Skip-gram with negative sampling (SGNS)](#skip-gram-with-negative-sampling-sgns)
- [CBOW](#cbow)
- [The negative sampler](#the-negative-sampler)
- [GloVe](#glove)
- [fastText and BPE](#fasttext-and-bpe)
- [Initialization, learning rate, and determinism](#initialization-learning-rate-and-determinism)
- [Hyperparameters and defaults](#hyperparameters-and-defaults)
- [Function surface](#function-surface)
- [Fixture and tolerances](#fixture-and-tolerances)
- [Failure modes](#failure-modes)
- [Implementation Examples](#implementation-examples)
- [Key Takeaways](#key-takeaways)
- [References](#references)
- [Crosswalk](#crosswalk)

## Notation and shapes

Word2Vec and GloVe learn **two** tables per model — the head topic needs only
one, because it treats a single trained matrix as given. The tables here are
stored **row-wise**, the opposite of the head's convention, because every
objective indexes a word's vector by row and takes an inner product; gathering
columns from a $(d \times V)$ matrix would be all transpose noise.

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $V$ | — | vocabulary size: the number of distinct tokens |
| $d$ | — | embedding dimension ($d = 8$ in the fixture; a few hundred in practice) |
| $W$ | $(V \times d)$ | the **centre** (input) table; row $W_w$ is centre word $w$'s vector |
| $C$ | $(V \times d)$ | the **context** (output) table; row $C_c$ is context word $c$'s vector |
| $\tilde{W}$ | $(V \times d)$ | GloVe's second (context-word) table |
| $b,\ \tilde{b}$ | $(V,)$ | GloVe's word and context-word bias vectors |
| $X$ | $(V \times V)$ | the integer co-occurrence matrix GloVe fits |
| $w$, $c$ | — | integer indices, $0 \le \cdot < V$ |
| $v_w$ | $(d,)$ | shorthand for the centre vector $W_w$ |
| $u_c$ | $(d,)$ | shorthand for the context vector $C_c$ |

The $(V \times d)$ tables use `float64`. The relationship to the head's matrix is
exact: the head's embedding matrix $E$, shape $(d \times V)$, is this topic's
trained centre table transposed,

$$E \;=\; W^{\top} \quad\text{(after training).}$$

So the head's "vectors, given" are literally this topic's output. SGNS and CBOW
share the pair of tables $W, C$; GloVe uses its own pair $W, \tilde{W}$ with
biases.

## Training pairs and the window

A corpus is a list of sentences, each a sequence of token indices. The
vocabulary is the **sorted** list of unique tokens, so `word_to_ix` is fixed by
the corpus alone and reproducible across machines. With window half-width $m$,
each position $i$ in a sentence of length $L$ pairs centre token $t_i$ with each
context token $t_{i+j}$ for

$$j \in \{-m, \dots, -1, 1, \dots, m\}, \qquad 0 \le i + j < L.$$

Pairs are emitted **by centre position $i$ ascending, then by offset $j$
ascending** — deterministic, and this fixes the order in which gradient descent
sees them. The window is applied to **one sentence at a time**, so no window
crosses a sentence boundary. The number of pairs a sentence yields is

$$\sum_{i=0}^{L-1}\bigl[\min(i, m) + \min(L - 1 - i,\, m)\bigr].$$

`training_pairs(sentence, window)` raises `ValueError` for `window < 1`.
`cooccurrence_matrix` reuses the same rule, so GloVe's counts and the
skip-gram/CBOW pairs come from one window definition.

## Skip-gram with negative sampling (SGNS)

For a centre–context pair $(w, c)$ with $v_w = W_w$, $u_c = C_c$, and $K$
negatives $n_1, \dots, n_K$ drawn from the [sampler](#the-negative-sampler), the
per-pair loss is

$$\ell(w, c, n_{1:K}) \;=\; -\log\sigma(v_w \cdot u_c) \;-\; \sum_{k=1}^{K} \log\sigma(-\,v_w \cdot u_{n_k}),$$

where $\sigma(z) = 1/(1 + e^{-z})$ and $u_{n_k} = C_{n_k}$. The positive term
rewards a high score for the true context; each negative term rewards a *low*
score for a sampled non-context. The total objective is the sum over all pairs
(Mikolov et al., 2013b). The gradients are the payload (Rong, 2014):

$$\frac{\partial \ell}{\partial v_w} \;=\; \bigl(\sigma(v_w\!\cdot\!u_c) - 1\bigr)\,u_c \;+\; \sum_{k} \sigma(v_w\!\cdot\!u_{n_k})\,u_{n_k},$$

$$\frac{\partial \ell}{\partial u_c} \;=\; \bigl(\sigma(v_w\!\cdot\!u_c) - 1\bigr)\,v_w, \qquad \frac{\partial \ell}{\partial u_{n_k}} \;=\; \sigma(v_w\!\cdot\!u_{n_k})\,v_w.$$

Only the rows a pair touches receive a gradient — $w$ in $W$, and $c$ and the
negatives in $C$ — so the update is **sparse**. Contributions accumulate one
index at a time (`+=`), so a negative that equals the context, or a repeated
negative, adds correctly rather than overwriting. The default $K = 2$ in the
fixture.

**Numerical stability.** The negative term is spelled $\log\sigma(-z)$, not
$\log(1 - \sigma(z))$; both losses are evaluated as $\log\sigma(z) =
-\operatorname{softplus}(-z)$ with $\operatorname{softplus}(x) =
\operatorname{logaddexp}(0, x)$, which returns the same value without the
underflow of a direct `log(sigmoid(z))`. The sigmoid itself multiplies through by
$e^{-\lvert z\rvert}$ so no intermediate exponential overflows.

## CBOW

CBOW reverses skip-gram: average the context vectors and predict the centre.
With context word indices $c_1, \dots, c_M$ (the window around $w$), let
$\bar{u} = \tfrac{1}{M}\sum_{j} C_{c_j}$. Negatives are scored against the
**centre** table, $v_{n_k} = W_{n_k}$:

$$\ell \;=\; -\log\sigma(\bar{u} \cdot v_w) \;-\; \sum_{k=1}^{K} \log\sigma(-\,\bar{u} \cdot v_{n_k}), \qquad v_w = W_w.$$

It is the SGNS loss with $\bar{u}$ in the centre's place and the centre and
negative words looked up in $W$. The gradients flow back through $\bar{u}$:

$$\frac{\partial \ell}{\partial \bar{u}} \;=\; \bigl(\sigma(\bar{u}\!\cdot\!v_w) - 1\bigr)\,v_w \;+\; \sum_{k} \sigma(\bar{u}\!\cdot\!v_{n_k})\,v_{n_k},$$

$$\frac{\partial \ell}{\partial W_w} = \bigl(\sigma(\bar{u}\!\cdot\!v_w) - 1\bigr)\,\bar{u}, \qquad \frac{\partial \ell}{\partial W_{n_k}} = \sigma(\bar{u}\!\cdot\!v_{n_k})\,\bar{u},$$

$$\frac{\partial \ell}{\partial C_{c_j}} \;=\; \frac{1}{M}\,\frac{\partial \ell}{\partial \bar{u}} \quad\text{for each context word } c_j.$$

The averaging splits the gradient on $\bar{u}$ evenly back to each of the $M$
context rows that formed it; a word that appears twice in the window collects the
$1/M$ share twice. Here $M$ is the number of context words actually passed, which
equals $2m$ in the interior of a sentence and is smaller at its edges. CBOW
shares the tables, the sampler, and the window with SGNS — only the scoring
differs.

## The negative sampler

Negatives are drawn i.i.d. from the **smoothed unigram distribution**

$$P(w) \;=\; \frac{f(w)^{\,\beta}}{\sum_{w'} f(w')^{\,\beta}}, \qquad \beta = \tfrac{3}{4},$$

where $f(w)$ is the raw corpus frequency of $w$ and $\beta$ is the `exponent`
parameter (default $0.75$). Raising a sub-unit power pulls the distribution
toward uniform, so **rare words are drawn as negatives more often than their raw
frequency** — the point being that negatives drawn in proportion to raw
frequency would be dominated by stop-words and carry little signal (Mikolov et
al., 2013b). `unigram_probabilities` returns a $(V,)$ vector that sums to $1$;
`sample_negatives(rng, probabilities, k)` is `rng.choice(V, size=k, p=P)`, kept on
the run's single generator so the whole run stays reproducible.

## GloVe

GloVe fits the **log co-occurrence counts** by weighted least squares. Build the
integer co-occurrence matrix $X$ ($X_{ij}$ = number of times $j$ falls in $i$'s
window), then minimize over the two tables $W, \tilde{W}$ and biases $b,
\tilde{b}$ (Pennington et al., 2014):

$$J \;=\; \sum_{i,\,j\,:\,X_{ij} > 0} f(X_{ij})\,\bigl(\underbrace{W_i^{\top}\tilde{W}_j + b_i + \tilde{b}_j - \log X_{ij}}_{r_{ij}}\bigr)^2,$$

$$f(x) \;=\; \min\!\left(1,\ (x / x_{\max})^{\alpha}\right), \qquad x_{\max} = 100,\ \ \alpha = \tfrac{3}{4}.$$

The weight grows with the count up to $x_{\max}$ and is capped at $1$ thereafter,
so very frequent pairs do not dominate the fit while rare pairs are not ignored.
Writing $r_{ij}$ for the residual, each observed entry contributes

$$\nabla_{W_i} = 2\,f(X_{ij})\,r_{ij}\,\tilde{W}_j, \qquad \nabla_{\tilde{W}_j} = 2\,f(X_{ij})\,r_{ij}\,W_i,$$

$$\nabla_{b_i} = 2\,f(X_{ij})\,r_{ij}, \qquad \nabla_{\tilde{b}_j} = 2\,f(X_{ij})\,r_{ij},$$

accumulated across every observed $(i, j)$. **The sum runs only over $X_{ij} >
0$** — a zero count contributes no term, which is also what keeps $\log 0$ out of
the objective. The learned vector for a word is the **sum of the two tables**,

$$e_i \;=\; W_i + \tilde{W}_i.$$

## fastText and BPE

Both are named here with their rule and not implemented; a subword implementation
is its own topic.

**fastText** (Bojanowski et al., 2017) represents a word as the sum of its
character $n$-gram vectors,

$$e_w \;=\; \sum_{g \,\in\, \mathcal{G}(w)} e_g,$$

where $\mathcal{G}(w)$ is the set of character $n$-grams of $w$ (plus the whole
word). Because the vector is built from pieces, an out-of-vocabulary word still
gets an embedding, which plain Word2Vec and GloVe cannot give.

**Byte-pair encoding (BPE)** (Sennrich et al., 2016) sidesteps the fixed
vocabulary differently: it merges frequent symbol pairs into subword units, so
the vocabulary is a learned set of word pieces rather than whole words. BPE and
its relatives are the tokenization layer under modern contextual models; the
[contextual embeddings](contextual-embeddings.md) topic carries that thread.

## Initialization, learning rate, and determinism

- **Initialization.** Every table and bias is drawn from $\mathcal{N}(0,
  \sigma^2)$ with $\sigma = 0.1 / \sqrt{d}$, from a single seeded
  `numpy.random.default_rng(seed)` in a fixed draw order: $W$ then $C$ for SGNS;
  $W$, $\tilde{W}$, $b$, $\tilde{b}$ for GloVe. The init is tiny but **never
  zero** — two identical rows would receive identical gradients forever and never
  separate — and the $1/\sqrt{d}$ scale keeps the initial inner products
  $O(0.01)$ regardless of $d$.
- **Learning rate.** Plain full-batch gradient descent at $\text{lr} = 0.05$:
  each epoch sums the loss and gradient over all pairs (SGNS/CBOW) or all observed
  entries (GloVe) and takes one step, $W \leftarrow W - \text{lr}\cdot\nabla W$.
  No optimizer, no momentum — the update rule is
  [`optimizer-update-rules`](optimizer-update-rules.md)'s subject,
  not this one's.
- **Determinism.** One generator drives init *and* sampling, in the documented
  order, so a run is reproducible to the bit. In `train_sgns` the $K$ negatives
  for each pair are drawn **once** and reused every epoch, which fixes the
  objective and makes the training loss a clean descent curve. (Classic Word2Vec
  resamples negatives each epoch; a teaching module that asserts a monotone loss
  on a toy corpus fixes the sample instead.) Each epoch's loss is recorded
  *before* that epoch's update.

## Hyperparameters and defaults

| Name | Symbol | Default | Role and typical range |
| --- | --- | --- | --- |
| `d` | $d$ | `8` | embedding dimension; a few hundred on real corpora |
| `window` | $m$ | `2` | context half-width; must be $\ge 1$ |
| `k` | $K$ | `2` | negatives per pair; 5–20 for small corpora, 2–5 for large (Mikolov et al., 2013b) |
| `exponent` | $\beta$ | `0.75` | sampler smoothing; $\beta = 1$ is raw frequency, $\beta = 0$ is uniform |
| `lr` | — | `0.05` | gradient-descent step size |
| `epochs` | — | `200` | full-batch passes |
| `x_max` | $x_{\max}$ | `100.0` | GloVe weight saturation count |
| `alpha` | $\alpha$ | `0.75` | GloVe weight ramp exponent |
| `seed` | — | `0` | seeds the single generator for init and sampling |

## Function surface

All names are re-exported from `dlhub.embeddings` (NumPy is always present, so the
re-export is eager). Full docstrings are on the generated
[API page](api/embeddings.md); this is the at-a-glance map.

| Function | Returns |
| --- | --- |
| `make_fixture()` | the planted bundle `{"corpus", "vocab", "word_to_ix", "sentences", "families", "dims"}` |
| `build_vocabulary(corpus)` | `(vocab, word_to_ix)` — the sorted vocabulary and its index map |
| `training_pairs(sentence, window=2)` | `list[(centre, context)]` for one sentence; raises `ValueError` if `window < 1` |
| `cooccurrence_matrix(sentences, vocab_size, window=2)` | the $(V \times V)$ `int64` matrix $X$ |
| `unigram_probabilities(sentences, vocab_size, exponent=0.75)` | the $(V,)$ sampler distribution $P$, summing to $1$ |
| `sample_negatives(rng, probabilities, k)` | a $(k,)$ array of negative indices |
| `sigmoid(z)` | $\sigma(z)$, computed without overflow |
| `sgns_loss(W, C, centre, context, negatives)` | the per-pair SGNS loss (`float`) |
| `sgns_gradients(W, C, centre, context, negatives)` | `{"W", "C"}` — full $(V \times d)$ gradient tables, zero off the touched rows |
| `cbow_loss(W, C, centre, context_words, negatives)` | the per-example CBOW loss (`float`) |
| `cbow_gradients(W, C, centre, context_words, negatives)` | `{"W", "C"}` |
| `glove_weight(x, x_max=100.0, alpha=0.75)` | the weight $f(x) \in (0, 1]$ |
| `glove_loss(W, W_tilde, b, b_tilde, X, x_max=100.0, alpha=0.75)` | the total GloVe loss $J$ (`float`) |
| `glove_gradients(W, W_tilde, b, b_tilde, X, x_max=100.0, alpha=0.75)` | `{"W", "Wt", "b", "bt"}` |
| `train_sgns(sentences, vocab_size, *, d=8, window=2, k=2, lr=0.05, epochs=200, seed=0, exponent=0.75)` | `{"W", "C", "history"}` |
| `train_glove(X, vocab_size, *, d=8, lr=0.05, epochs=200, seed=0, x_max=100.0, alpha=0.75)` | `{"W", "Wt", "b", "bt", "vectors", "history"}` |
| `nearest_neighbor(embeddings, query, k=1)` | the `k` nearest indices by cosine, query excluded, ties by ascending index; raises `ValueError` if `query` is out of range or `k < 1` |

The parameter is spelled `centre` (and `W_tilde`, `b_tilde`) in the signatures;
the GloVe return keys abbreviate to `"Wt"` and `"bt"`.

## Fixture and tolerances

The implementation and its tests share one **planted** toy corpus, built by
`make_fixture()`. It is hand-written, never downloaded — committed as a literal,
so it cannot drift — and every frequency and co-occurrence count is known by
construction, which makes the sampler's distribution and $X$ hand-checkable.

- **Corpus:** **12 sentences $\times$ 3 tokens = 36 tokens**, three sentences
  from each of four semantic families of four words each. For a family
  $[a, b, c, d]$ the three sentences are $[a, b, c]$, $[a, b, d]$, $[b, c, d]$ —
  so the plant is symmetric across families.
- **Vocabulary:** $V = 16$, the four families. Words are listed below in the
  **generating order** $[a, b, c, d]$ that forms the sentences; `word_to_ix`
  indexes the **globally sorted** vocabulary, so a word's integer index differs
  from its position here.

  | Family | $a$ | $b$ (hub) | $c$ | $d$ |
  | --- | --- | --- | --- | --- |
  | pets | `cat` | `dog` | `fish` | `bird` |
  | food | `bread` | `meat` | `seed` | `corn` |
  | actions | `chase` | `eat` | `feed` | `catch` |
  | places | `home` | `yard` | `pond` | `field` |

  Each sentence stays inside one family, so co-occurrence is **dense within a
  family and zero across families** — the structure the learned neighbors
  recover.

**Known answers (the correctness anchors):**

- **Training pairs.** Each length-3 sentence yields every ordered pair of its
  distinct positions — $3 \times 2 = 6$ pairs — so the corpus has
  $12 \times 6 = 72$ pairs at `window = 2`.
- **Co-occurrence (integer-exact).** $X$ is zero across families and dense
  within each — block-diagonal once words are grouped by family (not in the
  globally sorted index order). Within any family $[a, b, c, d]$ the block is
  identical up to relabeling; for `pets` ($a=$`cat`, $b=$`dog`, $c=$`fish`,
  $d=$`bird`):

  | $X$ | `cat` | `dog` | `fish` | `bird` |
  | --- | --- | --- | --- | --- |
  | `cat` | 0 | 2 | 1 | 1 |
  | `dog` | 2 | 0 | 2 | 2 |
  | `fish` | 1 | 2 | 0 | 1 |
  | `bird` | 1 | 2 | 1 | 0 |

  The family's second word (`dog`, `meat`, `eat`, `yard`) is the hub, co-occurring
  twice with each of the other three; the first word (`cat`, …) co-occurs twice
  with the hub and once with each of the remaining two.
- **Sampler (exact).** Raw frequencies are closed-form: the four hub words have
  $f = 3$, the other twelve have $f = 2$. So with $\beta = 3/4$ and
  $Z = 4\cdot 3^{3/4} + 12\cdot 2^{3/4} \approx 29.30$,

  $$P(\text{hub}) = \frac{3^{3/4}}{Z} \approx 0.0778, \qquad P(\text{other}) = \frac{2^{3/4}}{Z} \approx 0.0574,$$

  checked against the hand table within `atol = 1e-12`.
- **Gradient checks (exact).** For SGNS, CBOW, and GloVe the analytic gradient
  matches finite differences (via the public
  [`dlhub.training.gradient_checking.gradient_check`](gradient-checking.md))
  with relative error $< 10^{-6}$, and a **flipped-sign** variant is **rejected**
  (relative error $> 0.9$) — a gradient check that passes for a wrong gradient
  checks nothing.
- **Loss decreases (smoke).** SGNS training loss strictly decreases each epoch on
  the fixture.
- **It learns (seeded ranking).** After `epochs = 200`, `seed = 0`, `d = 8`,
  `window = 2`, `k = 2`, `lr = 0.05`, the nearest neighbor of `cat` by cosine
  (self excluded) is `dog`, and `cat`'s top-3 are exactly the three other pets.
  Asserted as a **rank**, never a cosine value, and only for the pets family — the
  plant is symmetric, so one family is a proof of concept.

**Tolerances:**

| Comparison | Tolerance |
| --- | --- |
| Gradient check | relative error $< 10^{-6}$; flipped sign rejected at $> 0.9$ |
| Sampler probabilities | `atol = 1e-12` |
| Co-occurrence matrix | integer-exact |
| Nearest-neighbor ranking | exact (ranks, not floats) |
| Loss decrease | strict `<` each epoch |

## Failure modes

| Symptom | Cause | Fix |
| --- | --- | --- |
| A gradient check passes even with a sign error | the check only confirmed the correct gradient, never exercised a wrong one | assert the **flipped-sign** variant is rejected (relative error $> 0.9$) as well |
| Training loss is noisy, not monotone | negatives resampled every epoch, so the objective moves | draw the negatives once and reuse them each epoch |
| `log sigmoid` underflows to $-\infty$ | computing `log(sigmoid(z))` directly | use $\log\sigma(z) = -\operatorname{softplus}(-z) = -\operatorname{logaddexp}(0, -z)$ |
| `sigmoid` overflows or returns `nan` | the textbook $e^{-z}$ spelling for large negative $z$ | multiply through by $e^{-\lvert z\rvert}$ so the exponential is bounded |
| GloVe loss is `inf` or `nan` | a zero count reached $\log X_{ij}$ | sum only over observed entries $X_{ij} > 0$ |
| Two words never separate; vectors stay identical | tables initialized to zeros or identical rows | initialize from $\mathcal{N}(0, \sigma^2)$, $\sigma = 0.1/\sqrt{d}$ |
| Negatives are almost all frequent words | sampling in proportion to raw frequency | raise the frequency to $\beta = 3/4$ to flatten toward uniform |
| A run differs between calls | several generators, a global seed, or an unfixed draw order | drive init and sampling from one `default_rng(seed)` in a fixed order |
| GloVe neighbors look wrong | only $W$ was read out | use the sum $W_i + \tilde{W}_i$ |
| Nearest neighbors reorder between runs | tied cosines ordered nondeterministically | break ties by ascending index |

## Implementation Examples

### Complete Implementations:
> #### **[Word2Vec and GloVe — from scratch (NumPy)](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/embeddings/word2vec_and_glove.py)** - The data layer (planted corpus, sorted vocabulary, windowed training pairs, co-occurrence matrix), the `unigram^0.75` sampler, and the three objectives with their exact gradients: SGNS, CBOW, and GloVe. Ships `train_sgns` and `train_glove` (seeded full-batch descent) and `nearest_neighbor`. Checked against the fixture: finite-difference gradient checks for all three objectives with the flipped-sign rejection, the exact sampler table, the integer-exact co-occurrence matrix, strictly decreasing SGNS loss, and the seeded `cat → dog` ranking.
> #### **[Word2Vec and GloVe — TensorFlow idiom port](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/tensorflow/embeddings/word2vec_and_glove.py)** - The SGNS objective in TensorFlow's idiom — `tf.keras.layers.Embedding` for the two tables, the framework's own `tf.nn.nce_loss`, and `tf.GradientTape` for the gradients — checked for parity against the NumPy SGNS loss and gradients on the shared fixture from the same seeded tables. Where `nce_loss` bakes in its own sampler, the port documents the mapping to the reference `unigram^0.75` and asserts parity on a fixed sampled batch fed to both.

## Key Takeaways

1. Both models learn **two $(V \times d)$ row tables**, not one — SGNS and CBOW share $W$ (centre) and $C$ (context); GloVe uses $W, \tilde{W}$ with biases. The head's $(d \times V)$ matrix is this topic's trained $W$ transposed, $E = W^{\top}$.
2. Training pairs come from a sliding window applied **per sentence**, emitted by centre position then offset ascending; the order is deterministic and never crosses a sentence boundary.
3. SGNS minimizes $-\log\sigma(v_w\!\cdot\!u_c) - \sum_k \log\sigma(-v_w\!\cdot\!u_{n_k})$; its gradient touches only the centre row, the context row, and the negative rows, so the update is sparse.
4. CBOW is SGNS with the context **averaged** into $\bar{u}$ and the roles swapped; the gradient on $\bar{u}$ splits evenly ($1/M$) back to the context rows.
5. Negatives are drawn from the smoothed unigram $P(w) \propto f(w)^{3/4}$, which flattens the distribution so rare words are sampled as negatives more often than their raw frequency.
6. GloVe is weighted least squares on $\log X_{ij}$ over **observed counts only**, with $f(x) = \min(1, (x/x_{\max})^{\alpha})$, $x_{\max} = 100$, $\alpha = 3/4$; the output vector is the sum $W_i + \tilde{W}_i$.
7. Determinism comes from one seeded generator driving init ($\mathcal{N}(0, (0.1/\sqrt{d})^2)$) and sampling in a fixed order, with the per-pair negatives drawn once and reused; plain SGD at $\text{lr} = 0.05$, no optimizer.
8. On the planted fixture ($V = 16$, 72 pairs) every answer is checkable: the $P(w)$ table to `atol = 1e-12`, $X$ integer-exact, all three gradients to relative error $< 10^{-6}$ with the flipped sign rejected, and the seeded `cat → dog` ranking exact.

## References

- **Mikolov, T., Chen, K., Corrado, G. & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space.**  
  [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781) – Word2Vec: the skip-gram and CBOW architectures.
- **Mikolov, T., Sutskever, I., Chen, K., Corrado, G. & Dean, J. (2013). Distributed Representations of Words and Phrases and their Compositionality. NeurIPS.**  
  [https://arxiv.org/abs/1310.4546](https://arxiv.org/abs/1310.4546) – Negative sampling, the $\text{unigram}^{3/4}$ sampler, and the $K$ guidance.
- **Pennington, J., Socher, R. & Manning, C. (2014). GloVe: Global Vectors for Word Representation. EMNLP.**  
  [https://nlp.stanford.edu/pubs/glove.pdf](https://nlp.stanford.edu/pubs/glove.pdf) – The weighted least-squares objective and the $f(x)$ weighting.
- **Rong, X. (2014). word2vec Parameter Learning Explained.**  
  [https://arxiv.org/abs/1411.2738](https://arxiv.org/abs/1411.2738) – The step-by-step derivation of the skip-gram and CBOW gradients stated here.
- **Bojanowski, P., Grave, E., Joulin, A. & Mikolov, T. (2017). Enriching Word Vectors with Subword Information. _Transactions of the ACL_ 5, 135–146.**  
  [https://arxiv.org/abs/1607.04606](https://arxiv.org/abs/1607.04606) – fastText: the character $n$-gram sum.
- **Sennrich, R., Haddow, B. & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. ACL.**  
  [https://arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909) – Byte-pair encoding, the subword tokenizer named here.
- **Ng, A. (2018). Deep Learning Specialization, Course 5 (Sequence Models), Week 2 — Natural Language Processing & Word Embeddings.**  
  [https://www.coursera.org/specializations/deep-learning](https://www.coursera.org/specializations/deep-learning) – The family's framing and the $(d \times V)$ notation this page departs from.

## Crosswalk

| Reader wants to | Go to |
| --- | --- |
| Learn it step by step | [Word2Vec and GloVe (notebook)](../tutorials/05-word2vec-and-glove.ipynb) |
| Do it in a project | [Train word vectors on your own corpus](../how-to/train-word-vectors.md) |
| Look up an equation or rule | Word2Vec and GloVe — *this page* |
| Understand why it works | [Word2Vec and GloVe](../explanation/word2vec-and-glove.md) |
| Learn by running it | [Word2Vec and GloVe (notebook)](../tutorials/05-word2vec-and-glove.ipynb) |
