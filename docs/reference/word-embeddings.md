# Word embeddings

This page is the lookup reference for *static* word embeddings: the embedding
matrix and the lookup identity, the geometry of word vectors (cosine similarity,
nearest neighbors, and the analogy / parallelogram rule), the from-scratch PCA
projection to two dimensions, the PyTorch storage convention, and the fixture and
tolerances the implementation is held to. It states the equations and rules and
makes no argument for them.

Notation follows Andrew Ng's Deep Learning Specialization, Course 5, Week 2
(Ng, 2018): the embedding matrix $E$ has shape $(d \times V)$ — $d$ features
(rows) by $V$ vocabulary (columns). This page treats the vectors as **given** —
a small, hand-planted toy matrix (see [Fixture and tolerances](#fixture-and-tolerances))
whose every answer is exact. How embeddings are *learned* (skip-gram, CBOW,
negative sampling, GloVe, fastText) is a separate topic, and so are bias geometry
and contextual embeddings. The reasoning behind the mechanics catalogued here —
why orthogonal one-hot vectors carry no similarity, why analogies are vector
arithmetic, and why averaging is order-blind — is in this topic's explanation
view, reachable from the [Crosswalk](#crosswalk).

## Contents

- [Notation and shapes](#notation-and-shapes)
- [The embedding matrix and the lookup identity](#the-embedding-matrix-and-the-lookup-identity)
- [Cosine similarity](#cosine-similarity)
- [Nearest neighbors](#nearest-neighbors)
- [Analogies and the parallelogram rule](#analogies-and-the-parallelogram-rule)
- [PCA projection to two dimensions](#pca-projection-to-two-dimensions)
- [The PyTorch convention](#the-pytorch-convention)
- [Fixture and tolerances](#fixture-and-tolerances)
- [Failure modes](#failure-modes)
- [Implementation Examples](#implementation-examples)
- [Key Takeaways](#key-takeaways)
- [References](#references)
- [Crosswalk](#crosswalk)

## Notation and shapes

| Symbol | Meaning |
| --- | --- |
| $V$ | vocabulary size: the number of distinct words |
| $d$ | embedding dimension: the number of learned features per word (a few hundred in practice; $d = 4$ in the fixture) |
| $E$ | the embedding matrix, shape $(d \times V)$ — one word per **column** (Ng's convention) |
| $i$ | a word's integer index, $0 \le i < V$ |
| $o_w$ | the one-hot column for the word at index $i$: a $(V,)$ vector, $1$ at $i$ and $0$ elsewhere |
| $e_w$ | the embedding of that word: a $(d,)$ vector, the $i$-th column $E_{:,\,i}$ |
| `vocab` | the word list, **sorted**, which fixes the column order of $E$ and the `word_to_ix` index map |

The sorted vocabulary makes the encoding deterministic across machines: the same
corpus yields the same column order, so every index, neighbor, and analogy answer
is reproducible. The fixture stores $E$ as `float64`.

## The embedding matrix and the lookup identity

For a word at index $i$, its embedding is the matrix–vector product of $E$ with
its one-hot column — which selects a single column:

$$e_w \;=\; E\,o_w \;=\; E_{:,\,i} \;\in\; \mathbb{R}^{d}.$$

This identity is the topic's spine: an **embedding layer is a bias-free linear
layer whose input is one-hot** (Bengio, Ducharme, Vincent & Jauvin, 2003). Because
$o_w$ is zero everywhere but one coordinate, the product reads off column $i$ and
nothing else, so no implementation ever materializes $E\,o_w$ — a **gather by
index** returns the same vector at a fraction of the cost. The reference exposes
both forms, and they agree exactly:

- `one_hot(i, V)` builds $o_w$;
- `lookup(E, i)` returns `E[:, i]` directly;
- `E @ one_hot(i, V) == lookup(E, i)` for every $i$ (the definition equals the
  implementation).

## Cosine similarity

Similarity between two word vectors $u, v \in \mathbb{R}^d$ is the cosine of the
angle between them — the dot product normalized by both magnitudes:

$$\cos(u, v) \;=\; \frac{u \cdot v}{\lVert u\rVert_2\,\lVert v\rVert_2}.$$

| Property | Value |
| --- | --- |
| Range | $[-1, 1]$ |
| Identical directions | $+1$ |
| Orthogonal | $0$ |
| Opposite directions | $-1$ |
| Scale | magnitude-independent (direction only) |
| Zero vector | returns `0.0` by convention — the cosine is undefined for a zero-norm vector, and `0.0` ("no similarity") is returned rather than a `nan` that would silently poison a ranking |

## Nearest neighbors

The nearest neighbors of a query index $i$, top-$k$:

1. score every other word $j$ by $\cos(E_{:,\,i}, E_{:,\,j})$;
2. sort by similarity **descending**;
3. **exclude the query itself**;
4. break ties by **ascending index**.

The result is deterministic: `nearest_neighbors(E, query, k)` returns a list of
`(index, cosine)` pairs, most similar first, of length $\min(k, V - 1)$. Asking
for more neighbors than exist clips $k$ to $V - 1$ (every word except the query)
rather than raising.

## Analogies and the parallelogram rule

An analogy "$a$ is to $b$ as $c$ is to $?$" is solved by **3CosAdd** (Mikolov,
Yih & Zweig, 2013): form the parallelogram target and return the word most
cosine-similar to it,

$$t \;=\; e_b - e_a + e_c, \qquad \text{answer} \;=\; \underset{w \,\notin\, \{a,\,b,\,c\}}{\arg\max}\;\cos(e_w,\, t).$$

Two rules make this exact and deterministic:

- **The three input words are excluded** from the candidates. Without this,
  `king − man + woman` returns `king`, because the subtraction leaves most of
  `king`'s mass in $t$ and `king` is its own nearest point.
- **Ties break by ascending index.**

This is the familiar "$\text{king} - \text{man} + \text{woman} \approx
\text{queen}$" written as $t = e_b - e_a + e_c$; the linear substructure that makes
it work is a property of the learned vectors (Pennington, Socher & Manning, 2014).
`analogy(E, a, b, c)` returns the answer's index.

The multiplicative variant **3CosMul** (Levy, Goldberg & Dagan, 2015) is more
robust on real corpora — it multiplies the three cosine terms instead of adding
them, which stops one large term from dominating — but it is not implemented here;
on the planted fixture the additive form already recovers the exact answer.

## PCA projection to two dimensions

To visualize $d$-dimensional vectors, project them onto their two
highest-variance directions with PCA, computed from scratch via the SVD. Given
$E$ of shape $(d \times V)$:

$$X = E^{\top}\;(V \times d,\ \text{one word per row}), \qquad X_c = X - \bar{x}, \qquad X_c = U\,\Sigma\,W^{\top},$$

where $\bar{x}$ is the mean word vector. The first two rows of $W^{\top}$ are the
top-2 principal directions, and the 2-D coordinates are the centered data
projected onto them:

$$Z \;=\; X_c\,W_{:,\,:2} \;\in\; \mathbb{R}^{V \times 2}.$$

**Sign convention (for determinism).** The SVD fixes each principal direction
only up to sign, so two runs could return mirror-image coordinates. Each component
is flipped so that its **largest-magnitude entry is positive**; the matching
coordinate column flips with it, so the reconstruction is unchanged and the output
is byte-for-byte reproducible.

`pca_project_2d(E)` returns a `PCAProjection` named tuple with fields `coords`
$(V \times 2)$, `components` $(2 \times d)$, `mean` $(d,)$, and `singular_values`
$(\min(V, d),)$ — the squared singular values being the variances explained by
PC1 and PC2. For data that genuinely lives in a 2-D plane (as the fixture does),
the projection is **loss-less** and recovers the plane exactly. **t-SNE** (van der
Maaten & Hinton, 2008) is the better tool for the nonlinear structure PCA cannot
capture; it is named here and not implemented.

## The PyTorch convention

PyTorch's `torch.nn.Embedding` stores its weight as $(V \times d)$ —
`num_embeddings` rows by `embedding_dim` columns — the **transpose** of Ng's
$(d \times V)$ matrix. So `weight == E.T`, and a word vector is a **row**,
`weight[i, :]`, not a column.

| This page | PyTorch |
| --- | --- |
| $E$, shape $(d \times V)$ | `weight`, shape $(V \times d)$, with `weight == E.T` |
| $e_w = E_{:,\,i}$ (a column) | `weight[i, :]` (a row) |
| `lookup(E, i)` / `E @ one_hot(i, V)` | `embedding(torch.tensor(i))` |

Load the reference matrix into a layer by transposing it once:

```python
import torch

# Ng writes E as (d, V); torch.nn.Embedding stores the transpose,
# so its weight has shape (V, d) -- one word per row.
weight = torch.tensor(E.T)  # (V, d)
embedding = torch.nn.Embedding.from_pretrained(weight)

# A word vector is therefore a row, not a column:
e_w = embedding(torch.tensor(i))  # == weight[i] == E[:, i]
```

A reader who conflates the two conventions transposes their matrix and gets $d$
"words" of length $V$. This same $(V \times d)$ weight is what gets **tied** to a
language model's output projection — the point where the embedding reference meets
the sequence-models family.

## Fixture and tolerances

The implementation and its tests share one **planted** toy embedding matrix,
built by `make_fixture()`. It is hand-written, not trained — committed as a
literal, never an external file, so it cannot drift — and every similarity,
neighbor, and analogy is exact and checkable by reading the numbers.

- **Dimensions:** $d = 4$, $V = 6$, dtype `float64`.
- **Latent axes:** coordinate 0 is *royalty*, coordinate 1 is *gender* ($+1$
  masculine, $-1$ feminine); coordinates 2–3 are $0$. The data therefore lives in
  a 2-D plane inside $\mathbb{R}^4$ (rank-2 by construction), which is what makes
  the PCA projection loss-less.
- **Vocabulary** (sorted, fixing column order and `word_to_ix`):

| Word | Index | Vector $(\text{royalty}, \text{gender}, 0, 0)$ |
| --- | --- | --- |
| `king` | 0 | $[2,\ 1,\ 0,\ 0]$ |
| `man` | 1 | $[0,\ 1,\ 0,\ 0]$ |
| `prince` | 2 | $[1,\ 1,\ 0,\ 0]$ |
| `princess` | 3 | $[1,\ -1,\ 0,\ 0]$ |
| `queen` | 4 | $[2,\ -1,\ 0,\ 0]$ |
| `woman` | 5 | $[0,\ -1,\ 0,\ 0]$ |

**Known answers (the correctness anchors):**

| Check | Result |
| --- | --- |
| Lookup identity | $E\,o_w = E_{:,\,i}$ exactly, for every $i$ |
| $\cos(\text{king}, \text{queen})$ | $3/5 = 0.6$ (royalty dominates gender, so a king is closer to a queen than to a man) |
| $\cos(\text{man}, \text{woman})$ | $-1.0$ (exact opposites on the gender axis) |
| Nearest neighbors (top-1) | `king→prince`, `queen→princess`, `prince→king`, `princess→queen`, `man→prince`, `woman→princess` (all strict, no ties) |
| Analogy `man : woman :: king : ?` | $t = e_{\text{woman}} - e_{\text{man}} + e_{\text{king}} = [2, -1, 0, 0] \Rightarrow$ **`queen`** ($\cos = 1.0$), beating runner-up `princess` ($\cos = 3/\sqrt{10} \approx 0.949$) |
| Analogy `prince : princess :: king : ?` | **`queen`** |
| PCA | loss-less (rank-2 data); gender is the higher-variance axis ($\sum x^2 = 6$ vs royalty $4$), so PC1 $\approx$ gender, PC2 $\approx$ royalty |

**Tolerances:**

| Comparison | Tolerance |
| --- | --- |
| Cosine and analogy | `atol = 1e-12` |
| PCA reconstruction error and pairwise-distance preservation | $< 10^{-10}$ |
| PCA output across repeated calls | byte-identical (determinism via the sign convention) |

## Failure modes

| Symptom | Cause | Fix |
| --- | --- | --- |
| An analogy returns one of its own input words | the inputs $\{a, b, c\}$ were not excluded from the candidates | exclude $\{a, b, c\}$ before the `argmax` |
| Every vector has length $V$, not $d$ | Ng's $(d \times V)$ matrix was fed to `nn.Embedding`, which expects $(V \times d)$ | transpose once: `weight = E.T` |
| `cosine_similarity` returns `nan` | one argument is the zero vector (zero norm) | return `0.0` for a zero-norm vector, as the reference does |
| Nearest neighbors differ between runs | tied cosines were ordered nondeterministically | break ties by ascending index |
| PCA coordinates mirror-flip between runs | the SVD's per-direction sign was left unfixed | flip each component so its largest-magnitude entry is positive |
| Indices or neighbors differ across machines | the vocabulary was built from an unordered set | build the vocabulary and `word_to_ix` from the **sorted** words |

## Implementation Examples

### Complete Implementations:
> #### **[Word embeddings — from scratch (NumPy)](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/embeddings/word_embeddings.py)** - The embedding matrix and lookup identity, cosine similarity with its zero-vector guard, deterministic nearest neighbors, 3CosAdd analogies with the inputs excluded, and the from-scratch PCA projection with its sign convention. Exposes `make_fixture`, `one_hot`, `lookup`, `cosine_similarity`, `nearest_neighbors`, `analogy`, and `pca_project_2d`, and is checked against the planted fixture: the lookup identity, the rational cosine anchors ($0.6$, $-1.0$), the full nearest-neighbor table, both exact analogies, and loss-less, byte-identical PCA.
> #### **[Word embeddings — PyTorch idiom port](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/pytorch/embeddings/word_embeddings.py)** - The same vectors loaded into a `torch.nn.Embedding`, showing PyTorch's $(V \times d)$ row convention: `weight` equals `E.T`, so a word vector is a *row* `weight[i, :]` rather than a column. An idiom track, not a parity port — it documents the transpose, the `from_pretrained` `freeze=True` default (the opposite of the trainable `nn.Embedding(V, d)` constructor), and weight tying as a forward pointer, and is checked on its own correctness: the gather equals the NumPy `lookup` exactly, for every word, because the fixture's `float64` dtype is preserved. Exposes `embedding_from_reference`, `lookup`, and `tied_readout`.

## Key Takeaways

1. An embedding layer is a bias-free linear layer applied to a one-hot input: $e_w = E\,o_w = E_{:,\,i}$, so a gather by index returns the same column the matrix product would, and the two forms agree exactly.
2. Cosine similarity, $\cos(u, v) = (u \cdot v) / (\lVert u\rVert_2 \lVert v\rVert_2)$, measures direction in $[-1, 1]$ independent of magnitude; a zero-norm vector returns `0.0` rather than `nan`.
3. Nearest neighbors are deterministic: rank other words by descending cosine, exclude the query, and break ties by ascending index.
4. Analogies use 3CosAdd — $t = e_b - e_a + e_c$, then $\arg\max_w \cos(e_w, t)$ over $w \notin \{a, b, c\}$; excluding the three inputs is what stops `king − man + woman` from returning `king`. The multiplicative 3CosMul is more robust on real corpora.
5. PCA projects to the two highest-variance directions via the SVD of the centered, row-wise data; fixing each component's sign (largest-magnitude entry positive) makes the output byte-identical, and rank-2 data projects without loss.
6. PyTorch stores the transpose: `nn.Embedding.weight` is $(V \times d)$ with `weight == E.T`, so a word vector is the row `weight[i, :]` — conflating the two conventions transposes the matrix.
7. One planted `float64` fixture ($d = 4$, $V = 6$) fixes every answer exactly: $\cos(\text{king}, \text{queen}) = 0.6$, $\cos(\text{man}, \text{woman}) = -1.0$, `man : woman :: king : queen`, with cosine/analogy checked to `atol = 1e-12` and PCA to $< 10^{-10}$.

## References

- **Ng, A. (2018). Deep Learning Specialization, Course 5 (Sequence Models), Week 2 — Natural Language Processing & Word Embeddings.**  
  [https://www.coursera.org/specializations/deep-learning](https://www.coursera.org/specializations/deep-learning) – Source of the $(d \times V)$ notation and the framing used here.
- **Bengio, Y., Ducharme, R., Vincent, P. & Jauvin, C. (2003). A Neural Probabilistic Language Model. _Journal of Machine Learning Research_ 3, 1137–1155.** – Learned distributed word-feature vectors: the embedding matrix as trained parameters.
- **Mikolov, T., Chen, K., Corrado, G. & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space.**  
  [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781) – Word2Vec; the origin of the vectors this page treats as given.
- **Mikolov, T., Yih, W. & Zweig, G. (2013). Linguistic Regularities in Continuous Space Word Representations. NAACL-HLT, 746–751.** – The analogy / parallelogram property this page's 3CosAdd rule computes.
- **Pennington, J., Socher, R. & Manning, C. (2014). GloVe: Global Vectors for Word Representation. EMNLP.**  
  [https://nlp.stanford.edu/pubs/glove.pdf](https://nlp.stanford.edu/pubs/glove.pdf) – The linear substructure that makes vector analogies work.
- **Levy, O., Goldberg, Y. & Dagan, I. (2015). Improving Distributional Similarity with Lessons Learned from Word Embeddings. _Transactions of the ACL_ 3, 211–225.** – 3CosMul, the multiplicative analogy scorer named here.
- **van der Maaten, L. & Hinton, G. (2008). Visualizing Data using t-SNE. _Journal of Machine Learning Research_ 9, 2579–2605.** – The nonlinear projection named as the better visualization tool.

## Crosswalk

| Reader wants to | Go to |
| --- | --- |
| Learn it step by step | [Word embeddings (notebook)](../tutorials/04-word-embeddings.ipynb) |
| Look up an equation or rule | Word embeddings — *this page* |
| Understand why it works | [Word embeddings](../explanation/word-embeddings.md) |
| Learn by running it | [Word embeddings (notebook)](../tutorials/04-word-embeddings.ipynb) |
