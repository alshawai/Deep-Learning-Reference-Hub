# Contextual embeddings

This page is the lookup reference for **contextual** and **text** embeddings: the
type-versus-token distinction that separates them from static vectors, the
four-part contract that a deployed embedding model actually is (encoder, pooling,
instruction prefix, normalisation), the retrieval regimes (bi-encoder,
cross-encoder, late interaction), a seven-row comparison of the method families,
MTEB's task taxonomy, and the boundary between what this material asserts and
what it cites. It states definitions, rules, shapes, and cited results, and makes
no argument for them.

The static baseline — one vector per word type, its geometry, and its analogy
arithmetic — is the sibling [word embeddings](word-embeddings.md) reference. How
those vectors are learned is [word2vec and GloVe](word2vec-and-glove.md); what
bias the geometry carries is [debiasing word embeddings](debiasing-word-embeddings.md).
This page picks the story up where the static assumption breaks: a word type no
longer has one vector.

The reasoning behind the mechanics catalogued here — why a token vector computed
in its sentence is a different object, why sentence vectors needed their own
training recipe, why retrieval split into two regimes — is in this topic's
explanation view, and the decision procedure for choosing a model for a task is
its how-to guide; both are reachable from the [Crosswalk](#crosswalk). The page
is code-free: the topic ships no implementation, and pretrained models are not
fixtures.

**Boundary.** This page owns *what a representation is and how it is consumed*.
How sequences are modelled — attention, state-space models, linear recurrence —
is [modern sequence models](modern-sequence-models.md); the internals of
attention and of pretraining objectives are deferred there and to the future
Transformer topics, and nothing here re-derives them.

## Contents

- [Notation and symbols](#notation-and-symbols)
- [Type and token](#type-and-token)
- [The four units of an embedding](#the-four-units-of-an-embedding)
- [The consumption contract](#the-consumption-contract)
- [Cosine on unit vectors](#cosine-on-unit-vectors)
- [The retrieval regimes](#the-retrieval-regimes)
- [Model families](#model-families)
- [Evaluation: MTEB's task taxonomy](#evaluation-mtebs-task-taxonomy)
- [The claims boundary](#the-claims-boundary)
- [Failure modes](#failure-modes)
- [Key Takeaways](#key-takeaways)
- [References](#references)
- [Crosswalk](#crosswalk)

## Notation and symbols

| Symbol | Meaning |
| --- | --- |
| $d$ | embedding dimension: the number of features per vector (a few hundred to a few thousand in deployed models) |
| $T$ | sequence length: the number of tokens in one input |
| $w_1, \dots, w_T$ | a tokenized input sequence; each $w_t$ is a **token occurrence**, not a word type |
| $f_\theta$ | the encoder, with parameters $\theta$ |
| $h_t$ | the contextual vector at position $t$, shape $(d,)$ |
| $e_w$ | the static embedding of word **type** $w$, shape $(d,)$ |
| $u, v$ | two vectors being compared |
| $\lVert v \rVert_2$ | the Euclidean norm $\sqrt{\sum_i v_i^2}$ |

Every vector in this topic is real-valued, shape $(d,)$. Token vectors are
emitted one per input position; a **text vector** (sentence, passage, document)
is one vector produced by pooling those token vectors, and is a different object
from any of them.

## Type and token

A **static** embedding is a function from a word **type** to a single vector:

$$w \;\mapsto\; e_w \;\in\; \mathbb{R}^{d}, \qquad e_w \text{ is the same in every sentence.}$$

A **contextual** embedding is a function from a **token occurrence** in a
sequence to a vector:

$$h_t \;=\; f_\theta(w_1, \dots, w_T)\big|_t \;\in\; \mathbb{R}^{d}.$$

The consequences, stated as rules:

- **There is no "the" vector for a word.** The river *bank* and the money
  *bank* have different vectors, because each is computed from the sequence it
  sits in.
- **A word type still has vectors, in the plural.** To obtain one vector per
  type from a contextual model you must choose an aggregation — usually the mean
  over occurrences — and report it; the aggregate is a derived statistic, not
  the model's output.
- **A sentence vector is not a contextual word vector.** The term *contextual
  embedding* is reserved for the per-token output. One vector for a whole input
  is a **text** embedding (sentence, passage, or document), produced by pooling
  (see [the consumption contract](#the-consumption-contract)).

The static object and its geometry — cosine similarity, nearest neighbours,
analogies — are defined in the sibling [word embeddings](word-embeddings.md)
reference and not restated here.

## The four units of an embedding

The `unit` column of the [model families](#model-families) table uses exactly
these four values.

| Unit | One vector represents | Produced by | Typical use |
| --- | --- | --- | --- |
| **word type** | every occurrence of a word, pooled away | static embeddings (Word2Vec, GloVe, fastText) | word similarity, analogies, a cheap baseline |
| **token** | one position in one input sequence | ELMo, BERT-family encoders | contextual analysis, token classification, the input to pooling |
| **sentence** | one short input (a sentence or a title) | SBERT, SimCSE | semantic similarity, clustering, deduplication |
| **passage** | one longer chunk (a paragraph or a document) | DPR, ColBERT, E5, BGE, Instructor | open-domain retrieval, RAG's retriever |

The unit is part of the model's identity, not a detail of its use: a sentence
vector and a token vector from the same encoder are not interchangeable, and a
model trained for one unit is not guaranteed to serve another well.

## The consumption contract

A deployed embedding is one artifact with four parts. Changing any part changes
the vectors, so an embedding result is reported as *(model, pooling,
normalisation)* together — and, for an instruction-tuned model, with the
instruction as a fourth.

| Part | What it fixes | Options | Consequence of changing it |
| --- | --- | --- | --- |
| **Encoder** | the contextual representation | an ELMo/BERT-family encoder, or a sentence-transformer variant | different vectors entirely |
| **Pooling** | how token vectors become one text vector | CLS token, mean, last token, learned pooling head | the same encoder under different pooling yields incompatible vectors |
| **Instruction prefix** | task conditioning of the input | none, or a task instruction prepended to the text (E5, Instructor) | the same text has different vectors under different instructions |
| **Normalisation** | the metric the vectors are compared under | none, or L2 to unit length | unnormalised vectors make cosine and inner product rank differently |

### Pooling options

| Pooling | Rule | Where it appears |
| --- | --- | --- |
| **CLS** | take the first special token's final hidden state | BERT-family sentence vectors; BGE's default |
| **Mean** | average the token vectors over the sequence | SBERT's default; E5 |
| **Last token** | take the final position's hidden state | decoder-style embedders |
| **Learned head** | a pooling layer trained with the embedding objective | model-specific |

Pooling is a choice, and the reference fixes it alongside the encoder for a
reason: two vectors called "BERT embeddings" can be incompatible if one is a CLS
vector and the other a mean vector, or if one was L2-normalised and the other
was not. There is no canonical sentence vector.

### Instruction prefixes

An instruction-tuned embedder takes the instruction as part of the input, so it
affects the vector rather than merely the query. E5 prefixes inputs with
`query:` and `passage:` and trains with those prefixes present (Wang et al.,
2022); Instructor is fine-tuned to follow a task instruction prepended to the
text (Su et al., 2022). The practical rule for a lookup: **record the prefix
with the model**, because a vector produced without it is a different vector
from the same weights.

## Cosine on unit vectors

Similarity between two vectors is the cosine of the angle between them, defined
in full under [cosine similarity](word-embeddings.md#cosine-similarity):

$$\cos(u, v) \;=\; \frac{u \cdot v}{\lVert u \rVert_2 \, \lVert v \rVert_2}.$$

After L2 normalisation, $\lVert u \rVert_2 = \lVert v \rVert_2 = 1$, so the
denominator is $1$ and

$$\cos(u, v) \;=\; u \cdot v.$$

This is why normalisation is one of the four contract parts rather than a
housekeeping step: on unit vectors, cosine and inner-product search return the
same ranking, and a vector store configured for one is correct for the other.
On unnormalised vectors they differ, and the difference is not a rescaling that
a threshold can absorb — the ranking itself changes.

Every similarity in this topic is cosine on unit vectors.

## The retrieval regimes

A retrieval score compares a query with a candidate. There are two regimes, and
a third that sits between them. Every retrieval claim names its regime, because
"dense retrieval works better" is meaningless without saying which one and on
what benchmark.

| Regime | How the pair is scored | Pre-computable? | Accuracy | Cost per query | Landmark |
| --- | --- | --- | --- | --- | --- |
| **Bi-encoder** | query and document are embedded **independently**; the score is the cosine or inner product of the two vectors | yes — document vectors are indexed offline | approximate: the encoder never sees the pair | one query encode, then an index lookup | DPR (Karpukhin et al., 2020) |
| **Cross-encoder** | query and document are **concatenated into one input** and scored jointly | no — the score depends on the pair | higher | one forward pass per candidate | the reranking half of the literature |
| **Late interaction** | both sides are encoded independently but **per-token** vectors are kept; the score matches token vectors after the fact | document token vectors are pre-computed (more storage than one vector) | between the two | a matching step per candidate | ColBERT (Khattab & Zaharia, 2020) |

The trade is the whole table: a bi-encoder buys pre-computation with accuracy,
a cross-encoder buys accuracy with a forward pass per candidate, and late
interaction buys token-level matching at the price of storing many vectors per
document. In a pipeline the usual arrangement is a bi-encoder for the first
stage and a cross-encoder as a reranker over a shortlist.

**Late interaction, stated exactly.** ColBERT keeps one vector per token on each
side and scores a pair with MaxSim — for each query token, its best match among
the document's tokens, summed:

$$\text{score}(q, d) \;=\; \sum_{i \,\in\, q} \; \max_{j \,\in\, d} \; q_i \cdot d_j .$$

**Out of domain, measure it.** BEIR, a heterogeneous zero-shot retrieval
benchmark, reported that dense retrievers which look strong in-domain can trail
a lexical baseline such as BM25 on unseen datasets (Thakur et al., 2021). That
result is the benchmark's reported finding at its own snapshot; it is the reason
[the claims boundary](#the-claims-boundary) refuses to treat an in-domain score
as transferable.

## Model families

Seven rows, one per era or family — not one per checkpoint, and not a
leaderboard. Cells carry a date and a citation, never a rank.

| Family (representatives) | Unit | Training signal | Pooling or output rule | What it is used for | Year + citation |
| --- | --- | --- | --- | --- | --- |
| **Static baselines** — Word2Vec, GloVe | word type | predict a word from its context (skip-gram, CBOW), or weighted least squares on log co-occurrence counts | one vector per type; no pooling | word similarity, analogy arithmetic, a cheap baseline | 2013–2014 (Mikolov et al., 2013; Pennington et al., 2014) |
| **ELMo** — biLSTM | token | two-layer bidirectional LSTM language model, pretrained on a corpus | one vector per position; the layers are combined with learned, task-specific weights | contextual features for downstream tagging, parsing, and classification | 2018 (Peters et al., 2018) |
| **BERT / RoBERTa** — Transformer encoder | token | masked-language-model pretraining; RoBERTa removes next-sentence prediction and trains longer on more data | one vector per position; a text vector requires a pooling choice (CLS or mean) | transfer to downstream tasks; contextual vectors as features | 2019 (Devlin et al., 2019; Liu et al., 2019) |
| **SBERT / SimCSE** — sentence embeddings | sentence | siamese and triplet fine-tuning on sentence pairs (SBERT); contrastive learning with dropout-generated positive pairs (SimCSE) | a fixed pooling (CLS or mean) chosen at training time, then L2 normalisation | semantic similarity, clustering, deduplication | 2019 / 2021 (Reimers & Gurevych, 2019; Gao et al., 2021) |
| **DPR / ColBERT** — retrieval | passage | contrastive training on question–passage pairs with in-batch negatives (DPR); late-interaction scoring trained pairwise (ColBERT) | DPR: one vector per passage and one per query. ColBERT: one vector per token, MaxSim | open-domain retrieval; the retriever under a RAG pipeline | 2020 (Karpukhin et al., 2020; Khattab & Zaharia, 2020) |
| **E5 / BGE / instruction-tuned** — modern embedders | sentence, passage | weakly-supervised contrastive pretraining on text pairs (E5); large-scale contrastive pretraining (BGE); instruction fine-tuning (Instructor) | mean pooling (E5), CLS pooling (BGE), and an instruction prefix that is part of the input | general-purpose embeddings for retrieval and similarity | 2022–2023 (Wang et al., 2022; Xiao et al., 2023; Su et al., 2022) |
| **Matryoshka** — truncatable embeddings | sentence, passage | wraps whatever base objective the pipeline already uses; a loss is applied at several nested prefix dimensions | one vector whose leading dimensions are usable on their own; truncation trades recall for storage | cheap first-stage search and reduced index size | 2022 (Kusupati et al., 2022) |

Read the table as a map of **recipes**, not checkpoints. A released checkpoint
carries its own pooling and normalisation, and those travel with the weights
rather than with the family row; the contract in
[the consumption contract](#the-consumption-contract) is what must be recorded
alongside any score.

Two historical notes the rows depend on:

- **The BERT row is really the BERT-family row.** RoBERTa's contribution is the
  finding that BERT's pretraining was undertrained, not a new representation
  (Liu et al., 2019).
- **The sentence rows exist because token vectors are not sentence vectors.**
  SBERT's point was that comparing two BERT sentence vectors directly is
  expensive and weak, and that a siamese fine-tune makes one forward pass per
  sentence enough (Reimers & Gurevych, 2019). SimCSE's point was that contrastive
  learning over positive pairs — including a sentence paired with itself under a
  different dropout mask — suffices (Gao et al., 2021).

## Evaluation: MTEB's task taxonomy

MTEB organizes embedding evaluation into eight task types (Muennighoff et al.,
2023). The taxonomy is the lookup: it fixes what "an embedding model is good
at" can mean, one task type at a time.

| Task type | What the score measures | Unit compared |
| --- | --- | --- |
| **Retrieval** | ranking a corpus against a query | query vs passage |
| **Reranking** | ordering an existing candidate list for a query | query vs passage |
| **Clustering** | grouping texts by their vectors | sentence |
| **Classification** | predicting a label from the vector alone | sentence |
| **STS** | correlation with human similarity judgements | sentence pair |
| **Pair classification** | a same/different label for two texts | sentence pair |
| **Summarization** | similarity of a summary to its source | sentence |
| **Bitext mining** | matching translation pairs across corpora | sentence |

Two rules follow from the taxonomy:

- **A single average hides the per-task spread.** An MTEB average is a mean over
  heterogeneous tasks, and a model can lead the average while trailing on the
  one task you have. Read the per-task score for your task type.
- **A benchmark score is one paper's reported result at one snapshot.** It is
  dated and attributed, never carried as a property of the model today.

## The claims boundary

The line between what this page asserts and what it merely reports, drawn so the
tables cannot be read as more than they say.

**Asserted** — definitional or architectural, standard, carried without a
per-paper citation:

- a static embedding is one vector per word type; a contextual encoder returns
  one vector per token occurrence;
- pooling is a choice, and different pooling over one encoder gives different
  text vectors;
- L2 normalisation makes cosine similarity equal the inner product;
- a bi-encoder is pre-computable and a cross-encoder is not; late interaction
  pre-computes per-token document vectors;
- attention is $O(T^2)$ compute in the sequence length (stated, not argued —
  its argument is [`modern-sequence-models`](modern-sequence-models.md)).

**Cited, dated, and attributed** — every empirical claim: every benchmark
result, every "outperforms", every MTEB and BEIR figure. Each is stated as the
paper's own reported result at the paper's scale. The
[References](#references) list is the only permitted citation base.

**Not entailed** — this page does **not** claim that:

- contextual embeddings remove social bias. They do not; the bias geometry and
  why geometric remedies are superficial are the
  [debiasing](../explanation/debiasing-word-embeddings.md) topic's subject.
- a high MTEB average guarantees performance on a particular task. It is an
  average over tasks, and the per-task spread is the usable signal.
- a sentence vector is canonical. Pooling and instruction are choices, so the
  vector is only defined together with them.
- "contextual beats static" licenses replacing a working pipeline without
  measuring it. The claim is about the **representation**, not about any
  particular deployment.

## Failure modes

| Symptom | Cause | Fix |
| --- | --- | --- |
| Two "BERT embeddings" are wildly dissimilar | different pooling (CLS vs mean), or one side normalised and the other not | pin *(model, pooling, normalisation)* for both sides, then re-embed |
| A query scores well against documents it should not match | query and corpus vectors came from different encoders, or one side omitted the instruction prefix | never mix encoders in one index; apply the prefix the model was trained with |
| Cosine and dot-product search return different neighbours | the vectors were not L2-normalised | normalise at index time and at query time |
| A word's "vector" looks like a sentence summary | a pooled text vector was read as a token vector | check the unit: token vectors come from the encoder, text vectors from pooling |
| A cross-encoder is too slow to serve as the retriever | a cross-encoder cannot pre-compute document vectors | retrieve with a bi-encoder, rerank a shortlist with the cross-encoder |
| An MTEB-average leader underperforms on the task at hand | a single average hides the per-task spread | read the per-task score for the task type in the [taxonomy](#evaluation-mtebs-task-taxonomy) |
| Dense retrieval trails BM25 on a new corpus | the out-of-domain brittleness BEIR reported | measure on your own corpus against a lexical baseline before committing |
| Debiasing a contextual model was expected to remove bias | contextual encoders do not remove social bias | see the [debiasing reference](debiasing-word-embeddings.md) |

## Key Takeaways

1. A static embedding is a function of a word **type**; a contextual embedding
   is a function of a **token occurrence**: $h_t = f_\theta(w_1, \dots,
   w_T)|_t$. There is no single vector for a word in a contextual model.
2. An embedding result is the triple *(model, pooling, normalisation)* — plus the
   instruction prefix for an instruction-tuned model. Changing any part yields
   incompatible vectors from the same weights.
3. The pooling options are CLS, mean, last token, and a learned head; there is
   no canonical sentence vector, and CLS and mean vectors from one encoder are
   not interchangeable.
4. After L2 normalisation, $\cos(u, v) = u \cdot v$: cosine and inner-product
   search rank identically on unit vectors and can rank differently without it.
5. Retrieval has two regimes with a third in between: a bi-encoder embeds
   independently and is pre-computable; a cross-encoder scores the pair jointly
   and is not; late interaction (ColBERT) keeps per-token vectors and scores
   with MaxSim. Every retrieval claim names its regime.
6. The seven-family table is a map of recipes — static, ELMo, BERT/RoBERTa,
   SBERT/SimCSE, DPR/ColBERT, modern embedders, Matryoshka — each dated and
   cited, none ranked.
7. MTEB's eight task types (retrieval, reranking, clustering, classification,
   STS, pair classification, summarization, bitext mining) fix what "good at
   embeddings" means; an average hides the per-task spread that decides a real
   choice.
8. Contextual embeddings do not remove social bias, an MTEB average is not a
   task guarantee, and a sentence vector is not canonical — the three
   non-entailments this page refuses to let its tables imply.

## References

- **Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K. & Zettlemoyer, L. (2018). Deep contextualized word representations. NAACL.**  
  [https://arxiv.org/abs/1802.05365](https://arxiv.org/abs/1802.05365) – ELMo; the contextual token representation and its learned layer combination.
- **Devlin, J., Chang, M.-W., Lee, K. & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL.**  
  [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805) – Contextual vectors from a Transformer encoder; masked-language-model pretraining.
- **Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L. & Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach.**  
  [https://arxiv.org/abs/1907.11692](https://arxiv.org/abs/1907.11692) – The "BERT was undertrained" result behind the family row.
- **Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP.**  
  [https://arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084) – The siamese/triplet recipe that made sentence vectors practical.
- **Gao, T., Yao, X. & Chen, D. (2021). SimCSE: Simple Contrastive Learning of Sentence Embeddings. EMNLP.**  
  [https://arxiv.org/abs/2104.08821](https://arxiv.org/abs/2104.08821) – Contrastive sentence embeddings; dropout as a positive pair.
- **Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D. & Yih, W. (2020). Dense Passage Retrieval for Open-Domain Question Answering. EMNLP.**  
  [https://arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906) – The bi-encoder retriever.
- **Khattab, O. & Zaharia, M. (2020). ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. SIGIR.**  
  [https://arxiv.org/abs/2004.12832](https://arxiv.org/abs/2004.12832) – Late interaction and MaxSim, between the bi-encoder and the cross-encoder.
- **Thakur, N., Reimers, N., Rücklé, A., Srivastava, A. & Gurevych, I. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. NeurIPS Datasets and Benchmarks.**  
  [https://arxiv.org/abs/2104.08663](https://arxiv.org/abs/2104.08663) – Zero-shot retrieval; the report that dense retrieval is brittle out of domain.
- **Muennighoff, N., Tazi, N., Magne, L. & Reimers, N. (2023). MTEB: Massive Text Embedding Benchmark. EACL.**  
  [https://arxiv.org/abs/2210.07316](https://arxiv.org/abs/2210.07316) – The eight-task-type taxonomy this page uses.
- **Wang, L. et al. (2022). Text Embeddings by Weakly-Supervised Contrastive Pre-training.**  
  [https://arxiv.org/abs/2212.03533](https://arxiv.org/abs/2212.03533) – E5; instruction-prefixed embeddings.
- **Xiao, S., Liu, Z., Zhang, P., Muennighoff, N., Lian, D. & Nie, J.-Y. (2024). C-Pack: Packed Resources For General Chinese Embeddings.**  
  [https://arxiv.org/abs/2309.07597](https://arxiv.org/abs/2309.07597) – BGE; the model family and its training recipe.
- **Kusupati, A. et al. (2022). Matryoshka Representation Learning. NeurIPS.**  
  [https://arxiv.org/abs/2205.13147](https://arxiv.org/abs/2205.13147) – Nested, truncatable embeddings.
- **Su, H. et al. (2022). One Embedder, Any Task: Instruction-Finetuned Text Embeddings.**  
  [https://arxiv.org/abs/2212.09741](https://arxiv.org/abs/2212.09741) – Instructor; task instructions as part of the input.
- **Mikolov, T., Chen, K., Corrado, G. & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space.**  
  [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781) – Word2Vec, the static row's predictive baseline.
- **Pennington, J., Socher, R. & Manning, C. (2014). GloVe: Global Vectors for Word Representation. EMNLP.**  
  [https://nlp.stanford.edu/pubs/glove.pdf](https://nlp.stanford.edu/pubs/glove.pdf) – The static row's count-based baseline.

## Crosswalk

| Reader wants to | Go to |
| --- | --- |
| Learn it step by step | not written for this topic — an explanation-first overlay; the hands-on lessons are the static-embedding notebooks ([04](../tutorials/04-word-embeddings.ipynb), [05](../tutorials/05-word2vec-and-glove.ipynb), [06](../tutorials/06-debiasing-word-embeddings.ipynb)) |
| Do it in a project | [Choose an embedding model](../how-to/choose-an-embedding-model.md) |
| Look up a model family, a pooling rule, or a benchmark | Contextual embeddings — *this page* |
| Understand why it works | [Contextual embeddings](../explanation/contextual-embeddings.md) |
| Learn by running it | not written for this topic — no implementation track, and pretrained models are not fixtures |

**Sibling topics (the Word Embeddings family):** the head
[word embeddings](word-embeddings.md) (what a word vector _is_),
[Word2Vec and GloVe](word2vec-and-glove.md) (how vectors are _learned_), and
[debiasing word embeddings](debiasing-word-embeddings.md) (what bias the geometry
carries). The sequence-architecture overlay is
[modern sequence models](modern-sequence-models.md).
