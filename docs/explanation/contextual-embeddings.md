# Contextual embeddings

A static embedding gives every occurrence of a word the same vector. That
single sentence is the whole limitation, and this page argues what changes when
it stops being true. The change is not a larger table of word vectors; it is a
different kind of object — a vector computed from the sentence rather than
looked up from the vocabulary — and once the object changes, everything a
consumer does with it changes too. Pooling stops being an afterthought and
becomes a decision that changes the answer. Sentence similarity stops falling
out of word vectors and becomes something a model has to be *trained* to
provide. Retrieval splits into two regimes with a real cost trade between them.
Evaluation becomes a spread of per-task scores rather than one number. And the
place embeddings are actually consumed moves from the input layer of a
classifier to the index at the centre of a search or generation pipeline.

It assumes the material of the sibling [word embeddings](word-embeddings.md)
page: word vectors as dense points, cosine similarity, nearest neighbours, and
the type-level geometry that makes analogy arithmetic work. How those static
vectors are learned from a corpus is
[Word2Vec and GloVe](word2vec-and-glove.md), and what bias their geometry
carries is [debiasing word embeddings](debiasing-word-embeddings.md); both are
taken as read. Nothing here is implemented or run — the topic is code-free by
design, and pretrained models are not fixtures — and for the same material as a
lookup table, or as a decision procedure for picking a model, see the
[Crosswalk](#crosswalk).

**Boundary.** This page owns *what a representation is and how it is consumed*.
How a sequence is modelled — attention, state-space models, linear recurrence —
belongs to [modern sequence models](modern-sequence-models.md), which argues why
attention displaced recurrence; the internals of attention and of pretraining
objectives are deferred there and to the future Transformer topics, and nothing
here re-derives them. Attention's $O(T^2)$ cost is used as a stated fact, not
argued.

A word on honesty first, because this is the fast-moving end of the family.
Everything here about *definitions and architecture* is a standard fact that
holds by construction: one vector per type versus one per occurrence, pooling
as a choice, a bi-encoder as pre-computable and a cross-encoder as not. Every
claim about *performance* is a specific paper's reported result at that paper's
scale, dated and attributed, never restated as a settled property of a model.
As of 2026 the model landscape in particular is a snapshot, and the page names
families with years rather than ranking checkpoints.

## Contents

- [The static ceiling](#the-static-ceiling)
- [A vector per occurrence: ELMo and BERT](#a-vector-per-occurrence-elmo-and-bert)
- [Pooling is a choice](#pooling-is-a-choice)
- [Why token vectors were not enough: SBERT and SimCSE](#why-token-vectors-were-not-enough-sbert-and-simcse)
- [Retrieval: two regimes and the trade between them](#retrieval-two-regimes-and-the-trade-between-them)
- [What the benchmarks measure, and what they do not](#what-the-benchmarks-measure-and-what-they-do-not)
- [The modern families, with years](#the-modern-families-with-years)
- [Where embeddings live now: retrieval and RAG](#where-embeddings-live-now-retrieval-and-rag)
- [What this does not entail](#what-this-does-not-entail)
- [Key Takeaways](#key-takeaways)
- [References](#references)
- [Crosswalk](#crosswalk)

## The static ceiling

A static embedding is a function from a word **type** to one vector: every
occurrence of *bank* reads the same row of the embedding matrix, whether the
sentence is about a river or a loan. The sibling page presents this as the
mechanism — a lookup by index, $e_w = E_{:,\,i}$ — and it is exactly the
mechanism that fails here. The word *bank* has at least two clearly distinct
senses, and the static table has one slot for both of them. The vector that
sits in that slot is not the river *bank* or the money *bank*; it is a
compromise between them and every other use, and it is closer to none of the
sense-specific neighbourhoods than a dedicated vector would be.

The tempting response is that more data fixes it. It does not, and the reason
is structural rather than statistical. A larger corpus makes the single row a
better estimate of the *aggregate* of the senses, which is to say it fits the
compromise more precisely. No amount of text can give one row two values. This
is the ceiling: not that static vectors are badly trained, but that the object
they are — a function of the type — cannot represent a distinction that lives
in the occurrence. The head page names this limit as the motivation for what
comes next, and it is the right place to see the geometry; here it is only the
problem statement.

Two qualifications keep the claim honest. First, *most* words are not
ambiguous in a way that hurts, and a static vector remains a strong, cheap
baseline — the ceiling is a property of the representation, not a verdict on
every pipeline that uses it. Second, the distinction that matters is not only
polysemy. The same word carries different *roles* depending on the sentence:
whether *it* refers to the bank or the loan, whether *good* is negated, whether
a token is part of a name. Static vectors have no slot for any of this.

## A vector per occurrence: ELMo and BERT

The fix is to change the function. A **contextual** embedding is a function
from a **token occurrence** — a position in a particular sequence — to a
vector:

$$h_t \;=\; f_\theta(w_1, \dots, w_T)\big|_t \;\in\; \mathbb{R}^{d}.$$

Here $w_1, \dots, w_T$ is the tokenized input, $\theta$ the encoder's
parameters, and $h_t$ the vector at position $t$, computed from *all* the
tokens in the sequence. The same word type in two different sentences is two
different calls to $f_\theta$ with different arguments, so it gets two
different vectors. There is no "the" vector for *bank* any more; there is the
vector for *bank* in this sentence.

ELMo (Peters et al., 2018) produced this with a two-layer bidirectional LSTM
language model — one forward and one backward pass over the sentence, each an
LSTM language model, with each position's representation built from both. The
paper's own contribution is not just the per-position vector but the
*combination* of the model's internal layers with learned, task-specific
weights — different downstream tasks were reported to prefer different depths. BERT (Devlin et al.,
2019) produced it with a Transformer encoder trained on two objectives: masked
language modelling, in which a fraction of the input tokens are replaced by a
mask and must be predicted from the rest, and next-sentence prediction. The
"bidirectional" in the name is about that pretraining: the encoder attends over
the whole sequence at once, rather than reading it strictly left to right — the
argument for why that architecture can do this at all is
[modern sequence models](modern-sequence-models.md)' subject, not this page's.

Two later corrections belong in the same breath, because they are how the
family is actually used. RoBERTa (Liu et al., 2019) reported that BERT's
pretraining was substantially undertrained — longer training, more data, larger
batches, and dropping next-sentence prediction all improved results — so "BERT"
in practice means a *family* of encoder pretraining recipes rather than one
checkpoint. And the biLSTM recipe itself gave way to the Transformer encoder
around the time the encoder family took over; the representation idea survived,
the architecture under it did not.

What changes for a consumer is worth stating plainly, because it is easy to
skim past:

- **There is no vocabulary table to precompute.** The unit of computation is a
  sequence, so obtaining any vector costs a forward pass. Whatever else a
  system does, that cost is now on the critical path.
- **The output is per token, not per word type.** To get one vector per type
  you must aggregate over occurrences and report how — the aggregate is a
  derived statistic, not the model's output.
- **The model's cost is quadratic in the sequence length** — attention is
  $O(T^2)$ — which is stated here and argued in
  [modern sequence models](modern-sequence-models.md). It is the reason long
  documents are chunked before they are embedded, a constraint that shapes
  every retrieval pipeline below.

## Pooling is a choice

A retrieval system, a classifier, or a deduplication job rarely wants a vector
per token. It wants one vector per sentence, passage, or document. That
requires a **pooling** rule, and the first thing to understand about pooling is
that it is a decision with consequences, not a formality.

The common rules are these. Take the **CLS token**: the first special token in
a BERT-style input, which the encoder lets attend to everything, so its final
hidden state is trained to summarise the sequence. Take the **mean** of all
token vectors. Take the **last token's** state, which is what decoder-style
embedders naturally expose. Or use a **learned pooling head** trained
alongside the embedding objective. Each produces a different vector from the
*same weights*, and the resulting spaces are not comparable with one another:
a CLS vector and a mean vector are different objects, and neither is a
canonical summary of the sentence.

Two further steps complete what a deployed model actually is. **Normalisation**
brings the vector to unit length, after which cosine similarity and inner
product coincide — $\cos(u, v) = u \cdot v$ — and the ranking an index returns
is the same under either metric. Skip it, and the two metrics can rank
differently, not merely on a different scale. And an **instruction prefix**,
for models trained with one, is prepended to the input: E5 prefixes queries
with `query:` and passages with `passage:` (Wang et al., 2022), and Instructor
is fine-tuned to follow a task instruction written into the input (Su et al.,
2022). An instruction changes the vector, which means the same text under two
instructions is two different objects — the instruction is part of the input,
not a comment on it.

So the artifact a system consumes is not "a model". It is a tuple: *(encoder,
pooling, normalisation)*, plus the instruction prefix when there is one. This
is the practical consequence of the whole page so far, and it is worth
restating as a rule: **two vectors called "BERT embeddings" can be mutually
incompatible**, and the incompatibility is invisible in the tensor shapes. Any
number reported about an embedding — a similarity, a retrieval score, a
benchmark result — is only meaningful with that tuple attached.

## Why token vectors were not enough: SBERT and SimCSE

Token-level pretraining makes a vector per occurrence meaningful. It does not
make a *sentence* vector good at comparing sentences, and the gap is easy to
miss because both objects share an encoder.

The problem is the training signal. Masked-language modelling teaches the
encoder to reconstruct tokens from context; nothing in that objective ever asks
two sentences that mean the same thing to land near each other. Mean- or
CLS-pooling is a *borrowed* heuristic: it assumes the encoder's geometry is
already arranged so that averaging preserves meaning, and it does not have to
be. Reimers and Gurevych (2019) reported exactly this — off-the-shelf BERT
sentence vectors, compared by cosine, performed *worse* than simply averaging
pretrained GloVe vectors on semantic similarity benchmarks. The representation
was contextual; the sentence-level geometry was not trained.

Sentence-BERT's response was to fine-tune the encoder with a **siamese**
structure: two copies of the network with shared weights encode two sentences,
and the pair is trained so that a similarity score computed from the pair's
vectors matches the target. The paper defines three objectives over that
structure — a **classification** objective whose softmax input is
$[u;\ v;\ \lvert u - v\rvert]$ (the one used for its NLI training), a
**regression** objective minimizing the squared error between the cosine
similarity and a label, and a **triplet** objective — and reports all three
outperforming the off-the-shelf baseline. The encoder then sees pairs at
training time and is explicitly asked to make the similarity correct. The cost argument
is the one that made this practical. Comparing every pair among 10,000
sentences by cross-encoding — putting both sentences into one forward pass and
scoring the pair — is roughly 50 million forward passes, which the paper
reports as around 65 hours on a GPU; with per-sentence vectors it took about
5 seconds. The numbers are the paper's, on its hardware, but the ratio is the
point: once each sentence is encoded once, similarity is a dot product.

SimCSE (Gao, Yao & Chen, 2021) moved the training signal to **contrastive
learning**, where the model is asked to pull a positive pair together and push
negatives apart, with the other sentences in the batch serving as negatives.
Its unsupervised version is the striking part: the positive pair is the same
sentence passed through the encoder twice with **different dropout masks**.
Nothing else changes — no labels, no augmentation — and the two masks give two
slightly different vectors, which the objective teaches the model to make
similar while the in-batch negatives keep it from collapsing every sentence to
one point. The supervised version uses natural-language-inference pairs:
entailment pairs are positives and contradiction pairs serve as hard negatives.
Both versions were reported to improve substantially on the off-the-shelf
baseline, and the dropout trick is the reason "unsupervised SimCSE" works at
all.

The general lesson outlives either paper. **A representation is good at what
its training signal asked it to be good at.** Token prediction asked for
context-sensitive token vectors; sentence comparison asks for a geometry in
which distances mean semantic distance. The second does not follow from the
first, which is why the sentence-embedding family exists as its own training
recipe rather than as a pooling option on a pretrained encoder.

## Retrieval: two regimes and the trade between them

Once sentences and passages have vectors, the obvious use is search: embed the
corpus, embed the query, return the nearest neighbours. That is where
embeddings are mostly consumed today, and it forces a design choice that the
word "embedding" hides.

A **bi-encoder** embeds the query and the document *independently* and compares
the two vectors. Because the document's vector does not depend on the query, it
can be computed once, offline, and stored in an index — so at query time the
work is one encoding plus a nearest-neighbour lookup. The price is accuracy:
the encoder never sees the pair, so it cannot use the interaction between query
and document to decide relevance. DPR (Karpukhin et al., 2020) is the landmark:
it trained a dual encoder with a contrastive objective on question–passage
pairs, using in-batch negatives, and reported large gains over BM25 on
open-domain question answering — on Natural Questions, top-20 retrieval
accuracy of 78.4% against BM25's 59.1%, in the paper's own evaluation.

A **cross-encoder** does the opposite. Query and document are concatenated into
one input and scored jointly by the full encoder, so every token of the query
can attend to every token of the document. It is considerably more accurate
and it is not pre-computable: the score depends on the pair, so there is one
forward pass per candidate, which is intractable against a large corpus. That
is why it appears as a **reranker** over a shortlist produced by a cheaper
first stage.

Between them sits **late interaction**, and ColBERT (Khattab & Zaharia, 2020)
is its canonical form. Both sides are encoded independently — so document
vectors are still pre-computable — but the model keeps a vector per token
rather than one per passage, and scores a pair with MaxSim: for each query
token, its best match among the document's tokens, summed over the query. That
buys token-level matching at the cost of storing many vectors per document, and
the paper reports it as orders of magnitude cheaper than a BERT cross-encoder
reranker — up to 170× faster and roughly four orders of magnitude fewer FLOPs —
while remaining competitive on quality. The three regimes are a spectrum, not a
contest: pre-computation, accuracy, and storage are the currencies, and every
pipeline spends them differently.

The warning that keeps this honest is BEIR (Thakur et al., 2021). Its
zero-shot evaluation across heterogeneous retrieval datasets reported that
BM25 — the lexical baseline — remained a strong competitor, and that dense
retrievers which looked excellent in-domain could fall behind it on datasets
they were not trained for. The finding is not that dense retrieval is bad; it
is that an in-domain score is not a transferable property. Any retrieval claim
has to name its regime *and* its benchmark, or it is comparing incommensurable
things.

## What the benchmarks measure, and what they do not

The benchmark that organises the modern landscape is MTEB (Muennighoff, Tazi,
Magne & Reimers, 2023), which evaluates embedding models across eight task
types — retrieval, reranking, clustering, classification, semantic textual
similarity, pair classification, summarization, and bitext mining — over 58
datasets in 112 languages in its initial release. The taxonomy is the useful
part: it makes "good at embeddings" a disjunction of concrete measurements
rather than a single word.

The trap is the average. A leaderboard number is a mean over tasks that make
different demands and are scored differently; a model can lead the average
while trailing badly on the one task a reader actually has. A model tuned for
asymmetric retrieval (short query, long passage) is not thereby tuned for
symmetric similarity between two sentences, and neither property follows from
clustering quality. The per-task score for *your* task type is the signal; the
average is a summary that can hide the decision-relevant fact.

Two more cautions, both about time. MTEB is a snapshot: models are added,
datasets are revised, and a reported position is a result as of that release,
not a property of the checkpoint today. And benchmark datasets leak into
training data — a model trained on the web has plausibly seen the test sets —
which is a reason to treat a surprising leaderboard lead as a question rather
than an answer. The honest use of a benchmark is to narrow the field, then to
measure on your own data.

## The modern families, with years

The models most people start from today are not the 2018–2019 landmarks but a
later generation of purpose-trained embedders. The page names them as families
with dates, because that is what the sources support; it does not rank them.

- **E5** (Wang et al., 2022) is trained with weakly supervised contrastive
  pretraining over large collections of text pairs, and its inputs carry the
  `query:` / `passage:` prefixes described above — the prefixes are part of the
  recipe, not an optional convenience.
- **BGE** (Xiao et al., 2023) is a family of embedding models released with its
  training recipe; the paper presents the package as a whole — data, training,
  and evaluation — rather than a single checkpoint.
- **Instruction-tuned embedders**, of which **Instructor** (Su et al., 2022) is
  the named example, take a task instruction as part of the input so that one
  model can serve several task types without retraining.
- **Matryoshka representation learning** (Kusupati et al., 2022) changes the
  vector's *storage* contract rather than its training signal: the objective is
  applied at several nested prefix dimensions, so the leading 64, 128, or 256
  dimensions are usable on their own. Truncation trades recall for index size,
  which is exactly the trade a large retrieval corpus wants to be able to make.

Read this list as recipes, not leaderboard rows. A released checkpoint carries
its own pooling, normalisation, and instruction convention; those travel with
the weights and are not inherited from the family name. The 2022–2023 dates are
the papers' publication dates, and as of 2026 the area has moved on further
than this page's source list covers — which is the correct reason to consult
the current documentation of whichever model you deploy rather than this
paragraph.

## Where embeddings live now: retrieval and RAG

The consumption pattern has shifted as much as the representation. In the
static era, word vectors were an *input layer*: you trained a classifier on
top of them and the embedding's job ended inside that model. Today the
dominant consumer is an **index** that outlives any single model, and the
embedding quality that matters is the quality of that index.

The pattern on top of it is **retrieval-augmented generation** (Lewis et al.,
2020). A retriever — in the paper, a dense bi-encoder, the DPR-style component
— selects passages from a corpus for the input, and a generator conditions on
both the query and the retrieved text. The paper's framing is that this
combines *parametric* memory, the weights of the generator, with
*non-parametric* memory, the document index, so knowledge can be updated by
editing the corpus rather than retraining the model. The embedding model is the
hinge of that design: it decides what counts as relevant before the generator
ever sees the text, so a retrieval failure is a generation failure with no way
to recover inside the generator.

It is worth being precise about where this page stops. Vector databases, ANN
indexes such as HNSW and IVF, product quantisation, chunking strategy, hybrid
search with lexical methods, and reranker engineering are all real and all
consequential — and all of them are *engineering* topics rather than
representation topics. The one claim this page makes about them is the
boundary: the embedding model sets the ceiling, and no index fixes a vector
that placed the right document nowhere near the query. A RAG engineering guide
is its own topic.

## What this does not entail

Four things follow from the material above, and four things very much do not.

- **Contextual embeddings do not remove social bias.** A vector computed in
  context is still computed from a corpus of human text, and it still absorbs
  the regularities in that text; nothing about the representation makes those
  regularities disappear. The geometry, the measurement, and why geometric
  remedies are superficial are the
  [debiasing topic](debiasing-word-embeddings.md)'s subject — the static
  measurements there (Bolukbasi et al., 2016) are not a claim about contextual
  models, and this page makes none of its own — the critique that geometric
  remedies are cosmetic (Gonen & Goldberg, 2019) is about the static geometry.
- **A high MTEB average is not a guarantee on your task.** It is a mean over
  heterogeneous tasks, and the spread is where the decision lives.
- **A sentence vector is not canonical.** The same encoder under CLS pooling,
  mean pooling, or no normalisation produces different and mutually
  incompatible vectors. The vector is only defined together with the tuple that
  produced it.
- **"Contextual beats static" is a claim about representation, not a licence.**
  It says a contextual encoder *can* represent a distinction that a static
  table cannot. Whether a particular deployment improves is an empirical
  question on that deployment's data, and the answer is sometimes no — the
  cost, the latency, and the domain shift can all eat the gain.

The summary that survives all four: the move from static to contextual changes
what an embedding *is*, and every downstream answer changes with it. The
practical discipline is the same one throughout — pin the artifact, name the
regime, date the benchmark, and measure on your own data.

## Key Takeaways

1. A static embedding is a function of a word **type** — one row per vocabulary
   entry, no matter how much data — so it cannot represent a distinction that
   lives in a particular occurrence of the word.
2. A contextual embedding is a function of a **token occurrence**,
   $h_t = f_\theta(w_1, \dots, w_T)|_t$, computed from the whole sequence. ELMo
   (2018) built it from a two-layer bidirectional LSTM language model; BERT (2019)
   from a Transformer encoder pretrained with masked language modelling, and
   RoBERTa (2019) showed the family was undertrained rather than finalized.
   This is the overlay the Deep Learning Specialization's Week 2 leaves
   implicit (Ng, 2018); the course's own words stop at static vectors.
3. The output is per token, so any sentence or passage vector requires
   **pooling** — CLS, mean, last token, or a learned head — and different
   pooling over the same weights gives incompatible vectors.
4. The consumed artifact is the tuple *(encoder, pooling, normalisation)*, plus
   the instruction prefix for models trained with one. Normalisation makes
   cosine and inner product coincide; an instruction prefix makes the same text
   two different vectors.
5. Token-level pretraining does not produce a sentence-level geometry. SBERT
   (2019) trained that geometry directly with siamese pairs and made
   per-sentence encoding practical; SimCSE (2021) showed contrastive learning
   with a dropout-generated positive pair is enough.
6. Retrieval has two regimes with a third in between. A **bi-encoder** (DPR,
   2020) embeds independently and pre-computes; a **cross-encoder** scores the
   pair jointly and cannot; **late interaction** (ColBERT, 2020) keeps per-token
   vectors and scores with MaxSim at a storage cost. Every retrieval claim
   names its regime and its benchmark — BEIR (2021) reported that dense
   retrieval's in-domain gains do not transfer for free.
7. MTEB (2023) fixes what "good at embeddings" means across eight task types;
   its average hides the per-task spread, and a score is a dated result, not a
   property of the checkpoint.
8. The modern families — E5, BGE, instruction-tuned embedders, Matryoshka — are
   recipes with years, consumed today mainly as the index under retrieval and
   RAG (Lewis et al., 2020). Contextual vectors still carry no removal of bias
   (see the [debiasing topic](debiasing-word-embeddings.md)), an average still
   is not a guarantee, and the sentence vector is still a choice.

## References

- **Peters, M., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K. & Zettlemoyer, L. (2018). Deep contextualized word representations. NAACL.**  
  [https://arxiv.org/abs/1802.05365](https://arxiv.org/abs/1802.05365) – ELMo; contextual token vectors from a two-layer biLSTM and their learned layer combination.
- **Devlin, J., Chang, M.-W., Lee, K. & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL.**  
  [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805) – Contextual vectors from a Transformer encoder; masked-language-model and next-sentence pretraining.
- **Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L. & Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach.**  
  [https://arxiv.org/abs/1907.11692](https://arxiv.org/abs/1907.11692) – The "BERT was undertrained" result behind the family reading.
- **Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP.**  
  [https://arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084) – The siamese/triplet recipe that made sentence vectors practical, and the encoding-cost argument.
- **Gao, T., Yao, X. & Chen, D. (2021). SimCSE: Simple Contrastive Learning of Sentence Embeddings. EMNLP.**  
  [https://arxiv.org/abs/2104.08821](https://arxiv.org/abs/2104.08821) – Contrastive sentence embeddings with dropout-generated positives.
- **Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D. & Yih, W. (2020). Dense Passage Retrieval for Open-Domain Question Answering. EMNLP.**  
  [https://arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906) – The bi-encoder retriever and its contrastive training.
- **Khattab, O. & Zaharia, M. (2020). ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. SIGIR.**  
  [https://arxiv.org/abs/2004.12832](https://arxiv.org/abs/2004.12832) – Late interaction, MaxSim, and its efficiency against a BERT cross-encoder.
- **Thakur, N., Reimers, N., Rücklé, A., Srivastava, A. & Gurevych, I. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. NeurIPS Datasets and Benchmarks.**  
  [https://arxiv.org/abs/2104.08663](https://arxiv.org/abs/2104.08663) – Zero-shot retrieval; why dense retrieval is brittle out of domain.
- **Muennighoff, N., Tazi, N., Magne, L. & Reimers, N. (2023). MTEB: Massive Text Embedding Benchmark. EACL.**  
  [https://arxiv.org/abs/2210.07316](https://arxiv.org/abs/2210.07316) – The eight-task taxonomy and the comparison this page declines to rank.
- **Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., Majumder, R. & Wei, F. (2022). Text Embeddings by Weakly-Supervised Contrastive Pre-training.**  
  [https://arxiv.org/abs/2212.03533](https://arxiv.org/abs/2212.03533) – E5; weakly supervised contrastive pretraining and the query/passage prefixes.
- **Xiao, S., Liu, Z., Zhang, P., Muennighoff, N., Lian, D. & Nie, J.-Y. (2024). C-Pack: Packed Resources For General Chinese Embeddings.**  
  [https://arxiv.org/abs/2309.07597](https://arxiv.org/abs/2309.07597) – BGE; the model family and its training recipe.
- **Su, H., Shi, W., Kasai, J., Wang, Y., Hu, Y., Ostendorf, M., Yih, W., Smith, N. A., Zettlemoyer, L. & Yu, T. (2022). One Embedder, Any Task: Instruction-Finetuned Text Embeddings.**  
  [https://arxiv.org/abs/2212.09741](https://arxiv.org/abs/2212.09741) – Instructor; the task instruction as part of the input.
- **Kusupati, A., Bhatt, G., Rege, A., Wallingford, M., Sinha, A., Ramanujan, V., Howard-Snyder, W., Chen, K., Kakade, S., Jain, P. & Farhadi, A. (2022). Matryoshka Representation Learning. NeurIPS.**  
  [https://arxiv.org/abs/2205.13147](https://arxiv.org/abs/2205.13147) – Nested, truncatable embeddings.
- **Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S. & Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.**  
  [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401) – RAG; parametric and non-parametric memory combined through a retriever.
- **Bolukbasi, T., Chang, K.-W., Zou, J., Saligrama, V. & Kalai, A. (2016). Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings. NeurIPS.**  
  [https://arxiv.org/abs/1607.06520](https://arxiv.org/abs/1607.06520) – The bias geometry the contextual shift does not remove.
- **Gonen, H. & Goldberg, Y. (2019). Lipstick on a Pig: Debiasing Methods Cover up Systematic Gender Biases in Word Embeddings. NAACL.**  
  [https://arxiv.org/abs/1903.03862](https://arxiv.org/abs/1903.03862) – Why geometric debiasing is superficial; see the [explanation](debiasing-word-embeddings.md).
- **Mikolov, T., Chen, K., Corrado, G. & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space.**  
  [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781) – Word2Vec, the static baseline this page moves away from.
- **Pennington, J., Socher, R. & Manning, C. (2014). GloVe: Global Vectors for Word Representation. EMNLP.**  
  [https://nlp.stanford.edu/pubs/glove.pdf](https://nlp.stanford.edu/pubs/glove.pdf) – The count-based static baseline.
- **Ng, A. (2018). Deep Learning Specialization, Course 5 (Sequence Models), Week 2.**  
  [https://www.coursera.org/specializations/deep-learning](https://www.coursera.org/specializations/deep-learning) – The framing this family of topics extends.

## Crosswalk

| Reader wants to | Go to |
| --- | --- |
| Do it in a project | [Choose an embedding model](../how-to/choose-an-embedding-model.md) |
| Look up a model family, a pooling rule, or a benchmark | [Contextual embeddings](../reference/contextual-embeddings.md) |
| Understand why it works | Contextual embeddings — *this page* |

**Sibling topics (the Word Embeddings family):** the head
[`word-embeddings`](../reference/word-embeddings.md) (what a word vector _is_),
[`word2vec-and-glove`](../reference/word2vec-and-glove.md) (how vectors are
_learned_), and [`debiasing-word-embeddings`](../explanation/debiasing-word-embeddings.md)
(what bias the geometry carries). The sequence-architecture overlay is
[`modern-sequence-models`](../explanation/modern-sequence-models.md).
