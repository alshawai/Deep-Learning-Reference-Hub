# How to train word vectors on your own corpus

You have a corpus — a pile of text you have already tokenized — and you want
static word vectors out of it: one fixed vector per word, with related words
landing near each other so that cosine similarity, nearest-neighbor lookups, and
the analogy rule all work. This guide takes you from raw tokens to trained,
inspected vectors using the hub's reference trainer, and it walks the knobs that
decide whether the vectors come out good — how you segment the corpus, skip-gram
versus CBOW, the context window, how many negatives to sample, the sampler's
`3/4` exponent, the embedding dimension, and the epoch count — with the default
and the trade-off for each.

It assumes you already know what a word vector *is* and what you read off one
(cosine similarity, nearest neighbors, the parallelogram analogy); that is the
word embeddings topic's job, and this guide links it where it leans on it. The
objectives and their gradients are stated in lookup form in this topic's reference
view, and *why* a predictive objective works and *why* the `3/4` exponent helps
are argued in its explanation view; both are reachable from the [Crosswalk](#crosswalk).

Everything below runs against the installed package
`dlhub.embeddings.word2vec_and_glove`, using the committed toy corpus as a stand-in
for yours. Swap in your own list of sentences and the same calls apply.

## Contents

- [Get your corpus into the trainer's shape](#get-your-corpus-into-the-trainers-shape)
- [Choose skip-gram or CBOW](#choose-skip-gram-or-cbow)
- [Set the context window](#set-the-context-window)
- [Choose how many negatives to sample](#choose-how-many-negatives-to-sample)
- [Tune the negative-sampling exponent](#tune-the-negative-sampling-exponent)
- [Pick a dimension, learning rate, and epoch count](#pick-a-dimension-learning-rate-and-epoch-count)
- [Train and read off the vectors](#train-and-read-off-the-vectors)
- [Fit GloVe when you have global counts](#fit-glove-when-you-have-global-counts)
- [Check that the vectors learned something](#check-that-the-vectors-learned-something)
- [Scale past the toy corpus](#scale-past-the-toy-corpus)
- [Troubleshoot a run that will not learn](#troubleshoot-a-run-that-will-not-learn)
- [Implementation Examples](#implementation-examples)
- [Key Takeaways](#key-takeaways)
- [References](#references)
- [Crosswalk](#crosswalk)

## Get your corpus into the trainer's shape

The trainer takes a **list of sentences**, each sentence a list of word tokens.
Two helpers turn that into what the objectives index: `build_vocabulary` assigns
every unique token an integer (sorted, so the index is fixed by the corpus alone
and not by the order words first appear), and you map each sentence to those
integers.

```python
from dlhub.embeddings.word2vec_and_glove import build_vocabulary, training_pairs

# Your corpus is a list of sentences, each a list of word tokens.
corpus = [["cat", "dog", "fish"], ["bread", "meat", "seed"]]

vocab, word_to_ix = build_vocabulary(corpus)  # vocab: the sorted unique tokens
sentences = [[word_to_ix[word] for word in sentence] for sentence in corpus]

# Each position pairs with its neighbors: (center, context) index pairs.
pairs = training_pairs(sentences[0], window=2)
```

The decision that matters here is **sentence segmentation**, because the context
window never crosses a sentence boundary — `training_pairs` runs on one sentence
at a time. If you hand the trainer one giant sentence, a word at the end of one
thought will be treated as context for a word at the start of the next. Split on
whatever boundary carries meaning for your text (sentences, paragraphs, documents,
log lines) before you build the vocabulary.

The hub ships a planted toy corpus so every example below is reproducible and its
answers are known by construction. It is 12 three-word sentences (36 tokens) over
a vocabulary of `V = 16`, four semantic families of four words each (pets, food,
actions, places), and every sentence stays inside one family — so co-occurrence is
dense within a family and zero across families. Reach it through `make_fixture`:

```python
from dlhub.embeddings.word2vec_and_glove import make_fixture

fixture = make_fixture()
sentences = fixture["sentences"]      # 12 sentences of token indices
vocab = fixture["vocab"]              # the sorted vocabulary, V = 16
word_to_ix = fixture["word_to_ix"]
V = fixture["dims"]["V"]
```

## Choose skip-gram or CBOW

Both objectives learn the same two tables — a **center** table `W` and a
**context** table `C`, each `(V, d)` with a word's vector on a row — and both draw
negatives from the same sampler. They differ only in what predicts what
(Mikolov et al., 2013a).

| | Skip-gram (SGNS) | CBOW |
| --- | --- | --- |
| Direction | center predicts each context word | averaged context predicts the center |
| Signal per example | one (center, context) pair | one center from its whole window |
| Rare words | learns them well — each gets its own updates | smoothed over by the averaging |
| Speed | slower (more updates per sentence) | faster |
| Hub trainer | `train_sgns` — the full loop | `cbow_loss` / `cbow_gradients` to drop into your own loop |

**Default to skip-gram with negative sampling.** It is the better choice on small
and medium corpora and for rare vocabulary, which is where a from-scratch run
usually lives, and it is the objective the hub ships as a ready training loop. The
module exposes CBOW as a loss-and-gradient pair that reverses the roles (average
the context, predict the center) so you can compose it into the same descent loop
when faster training on a large corpus is the priority; the sampler, the pairs,
and the negative draws are shared.

## Set the context window

The window half-width `m` is how far from the center word a context word may sit.
`training_pairs(sentence, window=m)` emits every in-sentence `(center, context)`
pair for offsets `-m … -1, 1 … m`, by center position then offset.

The window is a **semantic knob, not just a count**:

- A **small window** (`m = 1`–`2`) ties a word to its immediate neighbors, which
  tends to group words that are *substitutable* for each other (syntactic, same
  part of speech).
- A **large window** (`m = 5`–`10`) reaches across the sentence, which groups
  words that are merely *topically related*.

The module default is `window=2`. On the toy corpus a window of 2 already spans a
whole three-word sentence, so every within-sentence pair is captured: 12 sentences
yield 72 training pairs. Start at 2 for syntactic-flavored neighbors and widen only
if your downstream use wants topical grouping. The exact pairing rule is pinned in
the reference view (see the [Crosswalk](#crosswalk)).

## Choose how many negatives to sample

Negative sampling replaces the full-vocabulary softmax with a cheap binary
decision: push the true context pair together, and push `K` randomly sampled
*non-context* words apart (Mikolov et al., 2013b). `K` is the count of those
negatives per positive pair, set with the `k` argument to `train_sgns`.

- **Small corpora: `K = 5`–`20`.** More negatives give each update more signal,
  which a small corpus needs.
- **Large corpora: `K = 2`–`5`.** The sheer number of pairs supplies the signal,
  so you can afford fewer negatives per pair and the cheaper step.

The toy fixture uses `k=2` because it is tiny. The cost of a step grows with `K`
(each negative is one extra inner product and one extra row update), so `K` is a
direct speed-versus-signal lever.

## Tune the negative-sampling exponent

Negatives are not drawn uniformly, nor in proportion to raw frequency. They come
from the **smoothed unigram distribution**

$$P(w) = \frac{f(w)^{\alpha}}{\sum_{w'} f(w')^{\alpha}}, \qquad \alpha = 0.75,$$

where `f(w)` is the corpus frequency of `w`. The exponent is the `exponent`
argument to `unigram_probabilities` (and to `train_sgns`), and `0.75` is the
Word2Vec default (Mikolov et al., 2013b). Raising a sub-unit power pulls the
distribution toward uniform, so **rare words are drawn as negatives more often
than their raw frequency would allow** — which is the point: negatives drawn in
proportion to raw frequency are dominated by stop-words and carry little signal.

You can see the flattening directly:

```python
from dlhub.embeddings.word2vec_and_glove import unigram_probabilities

raw = unigram_probabilities(sentences, V, exponent=1.0)        # raw frequency
smoothed = unigram_probabilities(sentences, V, exponent=0.75)  # the default

dog, cat = word_to_ix["dog"], word_to_ix["cat"]
print(raw[dog] / raw[cat])        # 1.5  -- 'dog' occurs 3x, 'cat' 2x
print(smoothed[dog] / smoothed[cat])  # ~1.36 -- the gap is pulled toward 1
```

Two guard rails: `exponent=1.0` recovers raw frequency (stop-words will dominate
the negatives), and `exponent=0.0` makes every word equally likely (frequency
ignored entirely). `0.75` sits deliberately between them; move it toward `0` if
your negatives are swamped by a few very frequent tokens. *Why* this particular
value helps is argued in the explanation view (see the [Crosswalk](#crosswalk)).

## Pick a dimension, learning rate, and epoch count

Three remaining knobs on `train_sgns`:

- **`d`, the embedding dimension.** The toy corpus uses `d=8`. Real corpora use
  `100`–`300`: enough capacity to separate senses, not so much that a modest
  corpus cannot fill it. Wider is more expressive and more prone to overfit a
  small corpus.
- **`epochs`, the number of full passes.** `200` on the toy corpus. Watch the loss
  history rather than guessing — stop when it flattens (see
  [Check that the vectors learned something](#check-that-the-vectors-learned-something)).
- **`lr`, the step size.** `0.05` here. The loop is plain full-batch gradient
  descent — no momentum, no Adam — because this module teaches *the objective*,
  not the update rule; if you need an adaptive optimizer, that lesson lives in the
  [optimizer update rules reference](../reference/optimizer-update-rules.md).

You do not set the initialization: the tables start from
$\mathcal{N}(0, (0.1/\sqrt{d})^2)$, tiny but **never zero**, because two identical
rows would receive identical gradients forever and never separate. A single seed
drives both the initialization and the negative draws, so a run is reproducible to
the bit — pass `seed` to pin it.

## Train and read off the vectors

Put it together. `train_sgns` returns the two trained tables and the per-epoch loss
`history`; your word vectors are the rows of the center table `W`.

```python
from dlhub.embeddings.word2vec_and_glove import make_fixture, train_sgns, nearest_neighbor

fixture = make_fixture()
sentences, vocab, word_to_ix = fixture["sentences"], fixture["vocab"], fixture["word_to_ix"]
V = fixture["dims"]["V"]

result = train_sgns(sentences, V, d=8, window=2, k=2, lr=0.05, epochs=200, seed=0)
W = result["W"]  # the (V, d) center table -- these are your vectors

cat = word_to_ix["cat"]
neighbors = nearest_neighbor(W, cat, k=3)
print([vocab[j] for j in neighbors])  # ['dog', 'fish', 'bird'] -- the pets, recovered
```

Two things to know about reading the result:

- **`W` is the matrix the head topic calls `E`, transposed.** The head stores an
  embedding matrix `E` of shape `(d, V)` (a word is a column); this trainer stores
  `W` as `(V, d)` (a word is a row) because the objectives index rows and take
  inner products. The relationship is `E == W.T` once training is done — so if a
  downstream tool wants the head's convention, hand it `W.T`.
- **`nearest_neighbor` ranks by cosine similarity**, the geometry defined in the
  [word-embeddings reference](../reference/word-embeddings.md). It returns ranks,
  self excluded — do not read the raw cosine as an absolute score, especially on a
  tiny corpus.

On the toy corpus the nearest neighbor of `cat` is `dog` and its top three are
exactly the other three pets: the planted within-family structure, *learned* from
the pairs rather than put there by hand.

## Fit GloVe when you have global counts

GloVe takes a different route to the same goal: instead of streaming pairs, it
fits the **log of the global co-occurrence counts** directly (Pennington et al.,
2014). Build the `(V, V)` count matrix once, then fit.

```python
from dlhub.embeddings.word2vec_and_glove import cooccurrence_matrix, train_glove

X = cooccurrence_matrix(sentences, V)  # (V, V) integer counts, symmetric on this corpus
glove = train_glove(X, V, d=8, lr=0.05, epochs=200, seed=0)
vectors = glove["vectors"]  # W + W_tilde -- GloVe's final (V, d) vectors
```

Three things that are specific to GloVe:

- **The final vector is the sum of the two tables**, `W + W_tilde`, returned ready
  as `glove["vectors"]`. This is a GloVe convention, not a bug — summing the word
  and context roles averages out some of their noise.
- **A weighting `f(X_ij)` caps the influence of frequent pairs.** It ramps up with
  the count and saturates at `x_max = 100`, so a handful of very common pairs
  cannot dominate the fit while rare pairs still contribute. Zero counts contribute
  no term at all (there is no `log 0`).
- **Reach for GloVe when the global counts are what you have** — the co-occurrence
  matrix is built once and reused, where skip-gram re-streams pairs. On a large
  corpus this is a real efficiency; on this 36-token toy corpus neither objective
  produces trustworthy *absolute* neighborhoods (GloVe's top neighbor of `cat` here
  is not even a pet), so treat the GloVe run as a demonstration of the mechanics,
  not of recovered structure. The clean neighbor recovery above is the skip-gram
  claim, on this corpus, at these settings.

## Check that the vectors learned something

Before you trust vectors, confirm the run did two things. Neither needs a held-out
set on a corpus this size.

1. **The loss went down and flattened.** `result["history"]` is the per-epoch total
   loss; it should be strictly decreasing and level off by the last epochs. On the
   toy corpus it falls from about `150` to about `40` over 200 epochs. A flat or
   rising curve means a broken step — go to
   [Troubleshoot](#troubleshoot-a-run-that-will-not-learn).
2. **A word whose neighbors you already know lands them.** Pick a word with an
   obvious partner in your corpus and check its rank:

```python
history = result["history"]
assert history[-1] < history[0]  # loss decreased

ranked = nearest_neighbor(W, word_to_ix["cat"], k=3)
assert ranked[0] == word_to_ix["dog"]  # the known partner ranks first
```

Assert a **rank** ("`dog` is the nearest neighbor of `cat`"), never a cosine
*value* — ranks are stable across seeds and dimensions, absolute cosines are not.
This is exactly what the module's own test pins, and only for one family: a single
known-answer check is a proof of concept, and the finite-difference gradient checks
have already proven the math, so a ranking failure localizes to the training loop or
the sampler rather than the objective.

## Scale past the toy corpus

This module is a teaching reference: a readable from-scratch objective on a corpus
small enough to hand-check, not a production trainer. When you move to a real
corpus, four things change, and three of them point at tools beyond this module:

- **Dimension and negatives grow with the data.** Move `d` to `100`–`300`, and on a
  genuinely large corpus drop `K` toward `2`–`5`.
- **Subsample the frequent words.** The original Word2Vec also randomly drops very
  frequent tokens before forming pairs, which the hub module does not implement
  (Mikolov et al., 2013b). On a real corpus this both speeds training and improves
  rare-word vectors.
- **Out-of-vocabulary words need subwords.** This trainer learns one vector per
  token in the vocabulary and has nothing to say about a word it never saw.
  fastText fixes that by summing character n-gram vectors, `e_w = \sum_g e_g`, so an
  unseen word still gets a vector from its pieces (Bojanowski et al., 2017). The
  equation and the tokenization story are covered in the explanation view
  (see the [Crosswalk](#crosswalk)).
- **The hyperparameters above are the ones that pay off.** A careful study found
  that the window, the number of negatives, and the smoothing exponent — the knobs
  in this guide — account for much of the gap once credited to the choice of
  algorithm (Levy, Goldberg & Dagan, 2015). Tune them before you switch objectives.

For production training on real corpora, reach for an established library
(Gensim's Word2Vec, or Facebook's fastText) rather than this reference loop. What
you learned here — the window, `K`, the exponent, the dimension — are the same
knobs those libraries expose.

## Troubleshoot a run that will not learn

| Symptom | First thing to check |
| --- | --- |
| Loss is flat or rising across epochs | A broken step or wrong sign. Run `python -m dlhub.embeddings.word2vec_and_glove` and the test suite — the gradients are finite-difference checked, so confirm the math first, then your loop. |
| Every word is near every other word | `d` too small or `epochs` too few for the vocabulary. Raise the dimension or train longer and watch the loss flatten. |
| Neighbors are dominated by stop-words | Lower the sampling `exponent` toward `0.75` or below, and raise `K` so each update sees more informative negatives. |
| Unrelated topics are mixed into one neighborhood | The window is too wide, or sentences were not segmented — a window that spills across a boundary ties unrelated words. Shrink `window`, and split the corpus into real sentences first. |
| Words from different topics co-occur that should not | You passed one flat stream instead of a list of sentences; the window crossed boundaries. Pass `sentences` as a list of lists. |
| Two runs give different vectors | Pass a fixed `seed`; one generator drives both init and sampling, so a pinned seed is byte-reproducible. |
| A word you expect is absent from `vocab` | It was never in the corpus (or was filtered out). This trainer has no out-of-vocabulary fallback — see [Scale past the toy corpus](#scale-past-the-toy-corpus). |

## Implementation Examples

### Complete Implementations:

> #### **[Word2Vec and GloVe — from scratch (NumPy)](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/embeddings/word2vec_and_glove.py)** - The planted corpus, the training-pair and co-occurrence builders, the `unigram^0.75` sampler, and the SGNS, CBOW, and GloVe objectives with their exact hand-derived gradients and seeded training loops. Every gradient is checked against central finite differences, and a seeded run recovers the planted pets family.
> #### **[Word2Vec — TensorFlow parity port](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/tensorflow/embeddings/word2vec_and_glove.py)** - The same skip-gram objective in TensorFlow's idiom — `tf.keras.layers.Embedding` for the two tables, the framework's own `tf.nn.nce_loss`, and `tf.GradientTape` for the gradients — checked for parity against the NumPy reference on the shared fixture and sampled batch.

Rendered signatures and docstrings for the NumPy module are in the generated
[word-embeddings API reference](../reference/api/embeddings.md).

## Key Takeaways

1. Get the corpus into a **list of sentences** before anything else: the context
   window never crosses a sentence boundary, so segmentation decides what counts as
   co-occurrence. `build_vocabulary` then fixes a word's index by sorting, so the
   mapping is reproducible.
2. **Default to skip-gram with negative sampling** (`train_sgns`); it suits small
   and rare-word-heavy corpora and ships as a ready loop. Switch to CBOW
   (`cbow_loss` / `cbow_gradients` in your own loop) only when faster training on a
   large corpus matters more than rare-word quality.
3. The **window** is a semantic knob: small (`1`–`2`) for substitutable, syntactic
   neighbors; large (`5`–`10`) for topical ones. The default is `2`.
4. **`K` negatives** trade signal for speed — `5`–`20` on small corpora, `2`–`5` on
   large — and they are drawn from the smoothed unigram sampler, not uniformly.
5. The sampler's **`3/4` exponent** flattens the frequency distribution so rare
   words are oversampled as negatives; `1.0` recovers raw frequency (stop-words
   dominate) and `0.0` is uniform.
6. Your vectors are the rows of **`W`**, which equals the head topic's `E`
   transposed (`E == W.T`). GloVe's vectors are the **sum** `W + W_tilde`, returned
   as `glove["vectors"]`.
7. **Verify by rank, not by cosine value**: confirm the loss strictly decreased and
   that a word with a known partner ranks it first. On the toy corpus `cat`'s
   nearest neighbor is `dog` and its top three are the other pets.
8. This is a **teaching reference**. For real corpora raise `d` to `100`–`300`,
   subsample frequent words, consider fastText subwords for out-of-vocabulary
   words, and tune the window, `K`, and exponent — the hyperparameters that pay off
   (Levy, Goldberg & Dagan, 2015) — before reaching for a production library.

## References

- **Mikolov, T., Chen, K., Corrado, G. & Dean, J. (2013a). Efficient Estimation of Word Representations in Vector Space.**  
  [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781) – Skip-gram and CBOW, the two predictive objectives.
- **Mikolov, T., Sutskever, I., Chen, K., Corrado, G. & Dean, J. (2013b). Distributed Representations of Words and Phrases and their Compositionality. NeurIPS.**  
  [https://arxiv.org/abs/1310.4546](https://arxiv.org/abs/1310.4546) – Negative sampling, the `unigram^0.75` sampler, and frequent-word subsampling.
- **Pennington, J., Socher, R. & Manning, C. (2014). GloVe: Global Vectors for Word Representation. EMNLP.**  
  [https://nlp.stanford.edu/pubs/glove.pdf](https://nlp.stanford.edu/pubs/glove.pdf) – The count-based weighted-least-squares objective and the `W + W_tilde` sum.
- **Levy, O., Goldberg, Y. & Dagan, I. (2015). Improving Distributional Similarity with Lessons Learned from Word Embeddings. _TACL_ 3, 211–225.** – The hyperparameters — window, negatives, smoothing — that account for much of the measured gap between methods.
- **Bojanowski, P., Grave, E., Joulin, A. & Mikolov, T. (2017). Enriching Word Vectors with Subword Information. _TACL_ 5, 135–146.**  
  [https://arxiv.org/abs/1607.04606](https://arxiv.org/abs/1607.04606) – fastText: summing character n-gram vectors to cover out-of-vocabulary words.
- **Ng, A. (2018). Deep Learning Specialization, Course 5 (Sequence Models), Week 2.**  
  [https://www.coursera.org/specializations/deep-learning](https://www.coursera.org/specializations/deep-learning) – The course framing and the `(d, V)` embedding-matrix notation this family follows.

## Crosswalk

| Reader wants to | Go to |
| --- | --- |
| Learn it step by step | [Word2Vec and GloVe (notebook)](../tutorials/05-word2vec-and-glove.ipynb) |
| Do it in a project | Train word vectors on your own corpus — *this page* |
| Look up an equation or rule | [Word2Vec and GloVe](../reference/word2vec-and-glove.md) |
| Understand why it works | [Word2Vec and GloVe](../explanation/word2vec-and-glove.md) |
| Learn by running it | [Word2Vec and GloVe (notebook)](../tutorials/05-word2vec-and-glove.ipynb) |
