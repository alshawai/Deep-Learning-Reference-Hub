# Word2Vec and GloVe

The [word embeddings](word-embeddings.md) page took the vectors as *given* and
showed what their geometry buys you — cosine similarity, nearest neighbours, and
the parallelogram analogy. It left one question open on purpose: where does that
geometry come from? Nothing makes *king* and *queen* sit near each other, or makes
"add royalty" a constant offset, unless some training objective puts them there.
This page argues the why behind the two objectives that do — Word2Vec's predictive
skip-gram and GloVe's count-based least squares — and the one result that shows
they are, underneath, the same thing.

It assumes the word embeddings topic's vocabulary: what a word vector is, cosine
similarity, nearest neighbours, the analogy rule, and the linear substructure that
makes analogies linear. It does not re-derive that geometry; it explains the
objectives that produce it. For the equations and shapes in lookup form see this
topic's [reference](../reference/word2vec-and-glove.md); for the hands-on version
see the how-to and the notebook through the [Crosswalk](#crosswalk). The notation
is the reference's: two tables, the centre table $W$ and the context table $C$,
each of shape $(V \times d)$ with a word's vector on a *row* ($v_w = W_w$, $u_c = C_c$)
— the transpose of the embedding matrix's $(d \times V)$ matrix $E$ (Ng, 2018),
because every objective below indexes rows and takes inner products.

## Contents

- [Two routes to the same vectors](#two-routes-to-the-same-vectors)
- [Why skip-gram predicts the context](#why-skip-gram-predicts-the-context)
- [Why negative sampling works](#why-negative-sampling-works)
- [Why the sampler uses the three-quarter power](#why-the-sampler-uses-the-three-quarter-power)
- [Why skip-gram and CBOW behave differently](#why-skip-gram-and-cbow-behave-differently)
- [Why GloVe fits and weights the log-counts](#why-glove-fits-and-weights-the-log-counts)
- [The Levy–Goldberg unification](#the-levygoldberg-unification)
- [Beyond whole words: fastText and BPE](#beyond-whole-words-fasttext-and-bpe)
- [Implementation Examples](#implementation-examples)
- [Key Takeaways](#key-takeaways)
- [References](#references)
- [Crosswalk](#crosswalk)

## Two routes to the same vectors

Both Word2Vec and GloVe rest on the **distributional hypothesis** — a word's
meaning is its pattern of use, "you shall know a word by the company it keeps"
(Harris, 1954; Firth, 1957). The head topic (word embeddings) took that as
motivation; here it becomes an objective, and there are two ways to turn "the
company it keeps" into numbers.

The **count-based** route writes the company down directly: build the
co-occurrence matrix $X$, whose entry $X_{ij}$ counts how often word $j$ falls in
word $i$'s window, and compress it into low-dimensional vectors. This is the old
tradition of latent semantic analysis (Deerwester et al., 1990) — the whole
corpus is reduced to one global table of counts, and the vectors are whatever
factorises that table best. GloVe is the modern member of the family.

The **predictive** route never forms the matrix. It slides a window over the
corpus and, at each position, asks a model to *predict* the words that actually
occur in the context from the word at the centre; the vectors are the by-product
of getting good at that game (Bengio et al., 2003; Collobert et al., 2011).
Word2Vec is the fast, stripped-down version of this idea. Because it streams one
window at a time and updates online, it scales to billions of tokens without ever
holding a $V \times V$ matrix in memory, which is most of why it displaced the
count-based methods when it appeared (Mikolov et al., 2013a).

For years these looked like rival philosophies — "predict, don't count" was a
genuine slogan. The surprise, which the [Levy–Goldberg
unification](#the-levygoldberg-unification) below makes precise, is that the
predictive route is *implicitly* performing the count-based route's
factorisation. The sections before it build each objective on its own terms; the
payoff is that they meet.

## Why skip-gram predicts the context

Skip-gram's model is deliberately simple. Give it a centre word $w$ and it should
assign high probability to the words that really appear around $w$. The textbook
way to turn scores into a probability over the whole vocabulary is the
**softmax**: score every candidate context word $c$ by the inner product
$v_w \cdot u_c$, then normalise,

$$P(c \mid w) \;=\; \frac{\exp(v_w \cdot u_c)}{\sum_{c'=1}^{V} \exp(v_w \cdot u_{c'})}.$$

Train this to maximise the probability of the observed context words and the
geometry falls out exactly as you would want: to raise $P(c \mid w)$ the model
must raise $v_w \cdot u_c$ relative to every *other* word, which drags the vectors
of words that share contexts together. It is the same per-step objective a neural
language model uses to predict the next word (Bengio et al., 2003); the only twist
is that skip-gram predicts each surrounding word independently rather than the one
next word.

The problem is the denominator. That sum runs over the entire vocabulary, so a
single step costs $O(V)$ — and with $V$ in the hundreds of thousands and billions
of windows to process, evaluating it for every pair is hopeless. Every version of
Word2Vec is, at bottom, a way to get the gradient of this objective *without*
paying for its normaliser. The 2013 papers offer two: a **hierarchical softmax**
that replaces the flat sum with a binary tree of about $\log_2 V$ decisions, and
**negative sampling**, which abandons the normalised probability altogether
(Mikolov et al., 2013b). The hierarchical softmax is a pure speed trick — a
Huffman tree over the vocabulary, worthwhile only as an optimisation — so the
lesson, and this topic's implementation, is negative sampling.

## Why negative sampling works

Negative sampling drops the goal of a calibrated distribution over the vocabulary
and replaces it with a far cheaper one: tell real context pairs apart from random
ones. For the observed pair $(w, c)$ it draws $K$ **negative** words
$n_1, \dots, n_K$ from a noise distribution and minimises

$$\ell(w, c, n_{1:K}) \;=\; -\log\sigma(v_w \cdot u_c) \;-\; \sum_{k=1}^{K} \log\sigma(-\,v_w \cdot u_{n_k}),$$

with $\sigma(z) = 1/(1 + e^{-z})$. Read it as $K + 1$ independent logistic
regressions that share the same vectors: one that should fire ($\sigma(v_w \cdot
u_c) \to 1$, the true pair) and $K$ that should not ($\sigma(v_w \cdot u_{n_k})
\to 0$, the sampled non-pairs). The gradient says the same thing — it drives
$\sigma(v_w \cdot u_c)$ toward $1$ and each $\sigma(v_w \cdot u_{n_k})$ toward $0$
(Rong, 2014) — and it touches only the $K + 2$ rows a pair mentions, so a step is
$O(K)$ instead of $O(V)$. With $K$ a small constant the normaliser is simply gone.

Why should pushing true pairs together and random pairs apart recover anything
meaningful? Because the positive term alone has a trivial, useless optimum: make
every vector enormous and aligned, and every inner product — true pair or not —
runs to $+\infty$. The negatives forbid that. A word is pulled toward the words it
genuinely co-occurs with *and* pushed away from a representative sample of
everything else, and the only way to satisfy both across the whole corpus is to
place each word near its real neighbours and far from unrelated ones — the very
geometry the head topic reads off. The contrast is the content: with no negatives
there is nothing to be near *relative to*.

This "classify against noise" move is **noise-contrastive estimation**, and
negative sampling is a simplified form of it that drops the terms which would keep
$\sigma(v_w \cdot u_c)$ a true probability (Mikolov et al., 2013b). That
simplification is exactly why the learned inner product is not a log-probability
but something slightly different — the quantity the [Levy–Goldberg
unification](#the-levygoldberg-unification) pins down.

## Why the sampler uses the three-quarter power

The negatives are drawn neither uniformly nor in proportion to raw frequency. They
come from the **smoothed unigram distribution**

$$P(n) \;=\; \frac{f(n)^{3/4}}{\sum_{n'} f(n')^{3/4}},$$

where $f(n)$ is the corpus frequency of word $n$. The exponent is the whole
subject of this section, because the two obvious choices both fail.

Draw negatives **in proportion to raw frequency** (exponent $1$) and the stream of
negatives is almost all stop-words — *the*, *of*, *and*. Those co-occur with
everything, so "push $w$ away from *the*" is a signal the model cannot really act
on, while genuinely unrelated rare words almost never appear as negatives and so
are never pushed away from anything: the negatives are frequent but uninformative.
Draw negatives **uniformly** (exponent $0$) and you overcorrect — the vast tail of
rare words now dominates the negatives in proportion to how many of them exist,
spending the model's effort separating $w$ from words it was never going to be
confused with.

The exponent $3/4$ interpolates. Raising a count to a sub-unit power compresses the
range — it pulls large frequencies down harder than small ones — so the
distribution flattens toward uniform without discarding frequency outright:
frequent words are still sampled more often, just not ruinously so, and rare words
appear often enough to be pushed away when they should be. On the planted fixture
the four "hub" words occur three times each and the other twelve twice; raw
frequency would sample a hub word $3/2 = 1.5$ times as often as an ordinary one,
whereas $3^{3/4} / 2^{3/4} \approx 1.36$ under the smoothed sampler — the gap pulled
toward $1$ but not erased. The value $3/4$ itself is **empirical**, not derived: it
is the exponent Mikolov et al. (2013b) found worked best across tasks, and later
work treats the sampler's smoothing as one more hyperparameter to tune rather than
a constant of nature (Levy et al., 2015).

## Why skip-gram and CBOW behave differently

CBOW — continuous bag of words — runs the same machinery backwards: it averages
the context vectors into $\bar{u} = \tfrac{1}{M}\sum_j u_{c_j}$ and predicts the
centre word from that average, scoring negatives against the centre table (Mikolov
et al., 2013a). The loss, the sampler, the window, and the two tables are shared
with skip-gram; only the direction of prediction changes. Yet the two behave
differently, and the difference comes entirely from the averaging.

Skip-gram makes **one training example per context word**: a centre word in a
window of $2m$ neighbours generates $2m$ separate $(w, c)$ pairs, each a clean,
undiluted update to one context word's vector. CBOW makes **one example per
window**: the $M$ context vectors are blended into a single $\bar{u}$, and the
gradient on $\bar{u}$ is split $1/M$ back to each of them. Averaging is a low-pass
filter — it keeps what the context words have in common and cancels what is
particular to any one of them.

That one difference sets the trade-off. A rare word carries its sharpest signal
when it is handled on its own, which is exactly what skip-gram does, so it learns
rare words and small corpora well at the cost of more updates per sentence. CBOW
dilutes a rare context word into an average dominated by its commoner neighbours,
which smooths that word's vector but trains faster and tends to give slightly
better vectors for *frequent* words, where the averaging is a feature rather than
a loss. The standard advice follows directly: reach for skip-gram with negative
sampling on small or rare-word-heavy corpora, and for CBOW when speed on a large
corpus matters more than the tail (Mikolov et al., 2013a). It is the familiar
bias–variance pull in new clothes — averaging lowers the variance of each update
and raises its bias.

## Why GloVe fits and weights the log-counts

GloVe returns to the global count matrix $X$, but it does not fit the counts
themselves. Pennington et al. (2014) begin from an observation about *ratios*.
Take two words, *ice* and *steam*, and a probe word $k$. The individual
probabilities $P(k \mid \textit{ice})$ and $P(k \mid \textit{steam})$ are hard to
read, but their ratio is sharp: for $k = \textit{solid}$ it is large, for
$k = \textit{gas}$ it is small, and for a $k$ that is irrelevant (*fashion*) or
common to both (*water*) it is near $1$. The *ratio* of co-occurrence
probabilities, not the raw count, is what carries meaning — so that is what the
vectors should encode.

Requiring vector arithmetic to reproduce those ratios leads, after their
derivation, to a log-bilinear form: the inner product plus two bias terms should
match the log of the count,

$$W_i^{\top}\tilde{W}_j + b_i + \tilde{b}_j \;\approx\; \log X_{ij}.$$

Why $\log$, and why the biases? Because a ratio of probabilities is a *difference*
of logs, so fitting $\log X_{ij}$ is what makes a difference of word vectors track
the log-ratio of co-occurrences — the same linear substructure the head topic
relied on for analogies. The biases $b_i, \tilde{b}_j$ absorb each word's overall
frequency, the part of $\log X_{ij}$ that says nothing about the *pair*;
subtracting them off leaves the vectors modelling the association rather than the
marginals.

One problem remains: a plain least-squares fit of this equation over all $(i, j)$
would weight a pair seen once as heavily as a pair seen a thousand times, and would
have to include the $X_{ij} = 0$ cells, where $\log 0$ is undefined. GloVe's weight

$$f(x) \;=\; \min\!\left(1,\ (x / x_{\max})^{\alpha}\right), \qquad x_{\max} = 100,\ \ \alpha = \tfrac{3}{4},$$

fixes both. At $x = 0$ the term never enters the sum — $f$ multiplies observed
cells only, so there is no $\log 0$. For small counts $f$ ramps up with $x$, so a
pair backed by more evidence counts for more. And beyond $x_{\max}$ it is **capped
at $1$**, so a handful of enormously frequent pairs — the stop-word co-occurrences
again — cannot dominate the fit the way an uncapped weight would let them. The cap
is GloVe's answer to the problem the $3/4$ sampler solves for skip-gram: do not let
the most frequent events drown out the rest. It is no accident that $\alpha$ is
$3/4$ too; both methods reach for the same sub-unit power to tame frequency.
GloVe's final vector for a word is the **sum of its two tables**,
$e_i = W_i + \tilde{W}_i$, which averages out some of the noise in either role.

## The Levy–Goldberg unification

Skip-gram and GloVe now look like opposites: one predicts and never builds a
matrix, the other builds the matrix and fits it. Levy & Goldberg (2014) showed the
opposition is an illusion — skip-gram with negative sampling is *implicitly*
factorising a matrix, and a recognisable one.

Fix a single pair $(w, c)$ and ask what value of the inner product
$x = v_w \cdot u_c$ the training drives it to. Collect every term in the global
objective that depends on $x$ — the positive term wherever $(w, c)$ is observed,
and a negative term wherever $c$ is sampled as a negative for $w$. Written as the
quantity SGNS *maximises* (the negative of the loss above), that per-pair
objective is

$$\#(w, c)\,\log\sigma(x) \;+\; K\,\frac{\#(w)\,\#(c)}{|D|}\,\log\sigma(-x),$$

where $\#(w, c)$ is how often the pair occurs, $\#(w)$ and $\#(c)$ are the single
counts, $|D|$ is the total number of pairs, and the negatives are drawn from the
unigram $\#(c)/|D|$. Differentiating with respect to $x$ and setting the result to
zero gives

$$x \;=\; v_w \cdot u_c \;=\; \log\!\left(\frac{\#(w, c)\,|D|}{\#(w)\,\#(c)}\right) - \log K \;=\; \mathrm{PMI}(w, c) - \log K.$$

The quantity $\log\frac{\#(w,c)\,|D|}{\#(w)\,\#(c)} = \log\frac{P(w,c)}{P(w)\,P(c)}$
is the **pointwise mutual information** of the pair — how much more often $w$ and
$c$ occur together than if they were independent. So at its optimum skip-gram sets
each inner product to the pair's PMI, shifted down by $\log K$. Stack those inner
products and the two tables satisfy $W C^{\top} = M$ with

$$M_{wc} \;=\; \mathrm{PMI}(w, c) - \log K,$$

the **shifted PMI matrix**. Learning $(V \times d)$ tables whose product
approximates $M$ is precisely a low-rank factorisation of $M$ — the predictive
model was factorising a count-derived matrix all along, never mind that it built
no matrix. (Because $\mathrm{PMI}$ is $-\infty$ for the many pairs that never
co-occur, the version one can factorise directly is the sparse **shifted positive**
PMI, $\max(M, 0)$; and the real sampler's $3/4$ smoothing shifts the target to a
*smoothed* PMI rather than the exact one above.)

This is the bridge to GloVe. GloVe fits
$W_i^{\top}\tilde{W}_j + b_i + \tilde{b}_j \approx \log X_{ij}$, and once the biases
absorb the two marginal frequencies $\log\#(i)$ and $\log\#(j)$, what the inner
product is left to model is $\log X_{ij} - \log\#(i) - \log\#(j) + \text{const}$ —
PMI again, up to an additive constant. Both objectives land on the same
linear-algebraic target: a low-rank factorisation of a (shifted) PMI matrix built
from co-occurrence counts. "Predict, don't count" and "count, don't predict" were
two routes to one matrix, which is why Levy et al. (2015) could later show the
measured gap between the methods is mostly a matter of hyperparameters — the
window, the number of negatives, the sampler's smoothing — rather than any deep
difference in what they compute.

It also closes the loop with the head topic. That page treated the vectors as
given and observed that their geometry carried the distributional structure of the
language — that directions and offsets were meaningful. The unification says *why*:
the inner products **are** (shifted, smoothed) pointwise mutual information, so the
"given structure" the head read off is exactly the co-occurrence statistics these
objectives factorise. The vectors are a compressed PMI matrix in a different
coordinate system.

## Beyond whole words: fastText and BPE

Everything so far learns **one vector per vocabulary word**, which has a hard
edge: a word the corpus never contained gets no vector at all, and two
morphological forms of one stem — *run*, *running*, *runner* — are unrelated atoms
that must each be learned from their own occurrences. **fastText** (Bojanowski et
al., 2017) removes the edge by representing a word as the sum of the vectors of its
character $n$-grams,

$$e_w \;=\; \sum_{g \,\in\, \mathcal{G}(w)} e_g,$$

where $\mathcal{G}(w)$ is the set of character $n$-grams of $w$ (plus the whole
word), and the same SGNS objective trains the $n$-gram vectors in place of word
vectors. Two things fall out for free: an out-of-vocabulary word still gets an
embedding, assembled from the pieces it shares with words that were seen; and
related forms, which share most of their $n$-grams, start out close because they
are literally built from overlapping parts. The cost is a larger parameter set and
the bookkeeping of an $n$-gram vocabulary.

**Byte-pair encoding (BPE)** (Sennrich et al., 2016) attacks the same
fixed-vocabulary limit from the tokeniser's side. Rather than deciding in advance
what a "word" is, it starts from characters and repeatedly merges the most frequent
adjacent pair into a new subword unit, stopping at a chosen vocabulary size; the
result is a vocabulary of word pieces in which any string, however unseen, is
spellable as a sequence of known units. That sidesteps out-of-vocabulary words
entirely, and it is the tokenisation layer underneath the contextual models that,
as of 2026, have largely replaced static word vectors for most tasks. The planned
*contextual embeddings* topic carries that thread; this topic ends where a word
still has exactly one vector.

## Implementation Examples

### Complete Implementations:
> #### **[Word2Vec and GloVe — from scratch (NumPy)](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/embeddings/word2vec_and_glove.py)** - The from-scratch teaching reference behind the arguments on this page: the planted corpus and its co-occurrence matrix, the `unigram^0.75` sampler, and the three objectives with their exact hand-derived gradients — SGNS, CBOW, and GloVe — plus seeded `train_sgns` / `train_glove` loops and `nearest_neighbor`. Every gradient is finite-difference checked (with the flipped-sign variant rejected), and a seeded run recovers the planted pets family.
> #### **[Word2Vec and GloVe — TensorFlow idiom port](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/tensorflow/embeddings/word2vec_and_glove.py)** - The SGNS objective in TensorFlow's idiom — `tf.keras.layers.Embedding` for the two tables, the framework's own `tf.nn.nce_loss`, and `tf.GradientTape` for the gradients — checked for parity against the NumPy SGNS loss and gradients on the shared fixture from the same seeded tables, with the mapping from `nce_loss`'s built-in sampler to the reference `unigram^0.75` documented and asserted on a fixed sampled batch.

## Key Takeaways

1. Both objectives exist to *put* the head topic's geometry into the vectors. The
   distributional hypothesis becomes a training objective in two ways: predict the
   context from the word (Word2Vec), or fit the global co-occurrence counts
   (GloVe).
2. Skip-gram's natural objective is a softmax over the whole vocabulary, which
   costs $O(V)$ per step. Negative sampling replaces it with $K + 1$ cheap logistic
   decisions — pull the true pair together, push $K$ sampled non-pairs apart — and
   the negatives are what prevent the trivial all-aligned collapse.
3. Negatives are drawn from the smoothed unigram $f(n)^{3/4}$ because raw frequency
   floods them with uninformative stop-words and uniform wastes effort on the rare
   tail; the $3/4$ power flattens between the two, and the exact value is empirical.
4. CBOW is skip-gram with the context averaged and the prediction reversed. The
   averaging is the only difference, and it makes CBOW faster but weaker on rare
   words — a bias–variance trade.
5. GloVe fits $\log X_{ij}$ with two tables and bias terms because it is *ratios*
   of co-occurrence that carry meaning; its weight $f(x)$ drops the $\log 0$ cells
   and caps frequent pairs at $x_{\max}$ — the same frequency-taming instinct as the
   $3/4$ sampler, down to the shared exponent.
6. Levy & Goldberg: at its optimum SGNS sets $v_w \cdot u_c = \mathrm{PMI}(w, c) -
   \log K$, so it implicitly factorises a shifted-PMI matrix; GloVe, with its biases
   absorbing the marginals, fits essentially the same PMI target.
7. The two families therefore share one target — a low-rank factorisation of a
   (shifted) PMI matrix — and their measured gap is mostly hyperparameters. This is
   also why the head topic's "given" geometry carries distributional structure: the
   inner products are mutual information.
8. fastText sums character-$n$-gram vectors so out-of-vocabulary and
   morphologically related words still get embeddings; BPE learns a subword
   vocabulary that spells any string — the tokeniser under the contextual models
   that, as of 2026, have largely superseded static vectors.

## References

- **Harris, Z. (1954). Distributional Structure. _Word_ 10(2–3), 146–162.** – The distributional hypothesis: a word's meaning is its pattern of use in context.
- **Firth, J. R. (1957). A synopsis of linguistic theory 1930–1955. In _Studies in Linguistic Analysis_. Blackwell.** – "You shall know a word by the company it keeps."
- **Deerwester, S., Dumais, S., Furnas, G., Landauer, T. & Harshman, R. (1990). Indexing by Latent Semantic Analysis. _JASIS_ 41(6), 391–407.** – Count-based distributional vectors, the tradition GloVe belongs to.
- **Bengio, Y., Ducharme, R., Vincent, P. & Jauvin, C. (2003). A Neural Probabilistic Language Model. _Journal of Machine Learning Research_ 3, 1137–1155.** – Learned distributed word vectors from a softmax language model.
- **Collobert, R., Weston, J., Bottou, L., Karlen, M., Kavukcuoglu, K. & Kuksa, P. (2011). Natural Language Processing (Almost) from Scratch. _Journal of Machine Learning Research_ 12, 2493–2537.** – Task-trained embeddings before Word2Vec.
- **Mikolov, T., Chen, K., Corrado, G. & Dean, J. (2013a). Efficient Estimation of Word Representations in Vector Space.**  
  [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781) – Word2Vec: the skip-gram and CBOW architectures.
- **Mikolov, T., Sutskever, I., Chen, K., Corrado, G. & Dean, J. (2013b). Distributed Representations of Words and Phrases and their Compositionality. NeurIPS.**  
  [https://arxiv.org/abs/1310.4546](https://arxiv.org/abs/1310.4546) – Negative sampling, the smoothed unigram sampler, and hierarchical softmax.
- **Pennington, J., Socher, R. & Manning, C. (2014). GloVe: Global Vectors for Word Representation. EMNLP.**  
  [https://nlp.stanford.edu/pubs/glove.pdf](https://nlp.stanford.edu/pubs/glove.pdf) – The ratio motivation, the weighted least-squares objective, and the $f(x)$ weighting.
- **Levy, O. & Goldberg, Y. (2014). Neural Word Embedding as Implicit Matrix Factorization. NeurIPS.** – SGNS implicitly factorises a shifted pointwise-mutual-information matrix.
- **Rong, X. (2014). word2vec Parameter Learning Explained.**  
  [https://arxiv.org/abs/1411.2738](https://arxiv.org/abs/1411.2738) – The step-by-step derivation of the skip-gram and CBOW gradients.
- **Levy, O., Goldberg, Y. & Dagan, I. (2015). Improving Distributional Similarity with Lessons Learned from Word Embeddings. _TACL_ 3, 211–225.** – Why the hyperparameters matter more than the choice of method.
- **Sennrich, R., Haddow, B. & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. ACL.**  
  [https://arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909) – Byte-pair encoding, the subword tokeniser named here.
- **Bojanowski, P., Grave, E., Joulin, A. & Mikolov, T. (2017). Enriching Word Vectors with Subword Information. _TACL_ 5, 135–146.**  
  [https://arxiv.org/abs/1607.04606](https://arxiv.org/abs/1607.04606) – fastText: the character $n$-gram sum.
- **Ng, A. (2018). Deep Learning Specialization, Course 5 (Sequence Models), Week 2.**  
  [https://www.coursera.org/specializations/deep-learning](https://www.coursera.org/specializations/deep-learning) – The family's framing and the $(d \times V)$ notation this topic transposes.

## Crosswalk

| Reader wants to | Go to |
| --- | --- |
| Learn it step by step | [Word2Vec and GloVe (notebook)](../tutorials/05-word2vec-and-glove.ipynb) |
| Do it in a project | [Train word vectors on your own corpus](../how-to/train-word-vectors.md) |
| Look up an equation or rule | [Word2Vec and GloVe](../reference/word2vec-and-glove.md) |
| Understand why it works | Word2Vec and GloVe — *this page* |
| Learn by running it | [Word2Vec and GloVe (notebook)](../tutorials/05-word2vec-and-glove.ipynb) |
