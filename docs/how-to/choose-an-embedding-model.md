# How to choose an embedding model

You have a task that consumes text similarity — semantic search or
retrieval-augmented generation (Lewis et al., 2020), clustering, deduplication,
classification, or
sentence similarity — and a model to pick for it. This guide goes from your
task, corpus, language, and budget to a defensible starting family, the
consumption contract you pin alongside it, and the measurement that decides
whether to keep it.

It decides **which model to consume**. That is a different altitude from the two
architecture guides: [choose a sequence model](choose-a-sequence-model.md)
picks an architecture family — attention, state-space, linear attention, or
recurrence — and [choose a gated cell](choose-a-gated-cell.md) picks the cell
inside a recurrent layer. Both assume you are designing and training a sequence
model. This guide assumes the opposite: the encoder is pretrained, and the
decisions are which one, configured how, and evaluated against what. How the
encoder models a sequence at all belongs to
[modern sequence models](../explanation/modern-sequence-models.md); here it is a
black box with a fixed interface. The families and pooling rules this guide
selects among are tabulated in this topic's reference view, and the reasons the
shift from static to contextual vectors happened are argued in its explanation
view; both are reachable from the [Crosswalk](#crosswalk).

A word on honesty before you start. As of 2026 this is a fast-moving area, and a
leaderboard position is a dated result on someone else's task list. This guide
separates what is **architectural** — pooling changes the vector, a bi-encoder
is pre-computable and a cross-encoder is not — from what is a **reported
result** from a named paper at a named scale. Empirical claims below are dated
and attributed. Nothing here promises that the top of a benchmark is the right
model for your data.

## Contents

- [Pin the consumption contract before you compare models](#pin-the-consumption-contract-before-you-compare-models)
- [Decide the regime before the model](#decide-the-regime-before-the-model)
- [Match the objective to the task](#match-the-objective-to-the-task)
- [Pick a starting family, and note its year](#pick-a-starting-family-and-note-its-year)
- [Match the model to language, domain, and length](#match-the-model-to-language-domain-and-length)
- [Count the dimension and the cost](#count-the-dimension-and-the-cost)
- [Measure before you commit](#measure-before-you-commit)
- [Decision table](#decision-table)
- [Diagnose an embedding model that underperforms](#diagnose-an-embedding-model-that-underperforms)
- [Key Takeaways](#key-takeaways)
- [References](#references)
- [Crosswalk](#crosswalk)

## Pin the consumption contract before you compare models

A deployed embedding is not a checkpoint; it is one artifact with four parts,
and every one of them changes the vector:

1. **Encoder** — an ELMo/BERT-family encoder, or a sentence-transformer variant.
2. **Pooling** — how per-token vectors become one vector: the CLS token (the
   convention for BERT-family sentence vectors and BGE), mean pooling (SBERT's
   default, and E5's), last-token (the convention of decoder-style embedders),
   or a learned pooling head.
3. **Instruction prefix** — an optional task instruction prepended to the input
   (E5 / Instructor style), which means the same text has different vectors
   under different instructions.
4. **Normalisation** — L2 to unit length, so **cosine** and inner product
   coincide.

Fix all four before you compare anything. Two checkpoints both called "BERT" can
produce vectors that are not comparable if their pooling or normalisation
differs, and the same text embedded under two different instructions is two
different vectors by design. An embedding result is therefore always reported as
*(model, pooling, normalisation)* together — plus the instruction, when one was
used — and never as a model name alone.

Every similarity in this topic is cosine on unit vectors, using the definition
pinned in the head's reference
([cosine similarity](../reference/word-embeddings.md#cosine-similarity)). If
your stored vectors are not normalised, inner-product search is not cosine
search, and a ranking that looks plausible can be measuring length instead of
direction.

## Decide the regime before the model

For any retrieval-shaped task, the first decision is not *which* encoder but
*which regime*. The two regimes trade accuracy against pre-computability, and
every retrieval claim names its regime — "dense retrieval works better" is
meaningless without saying which one and on what benchmark.

| Regime | What the encoder sees | Pre-computable? | Where it fits |
| --- | --- | --- | --- |
| **Bi-encoder** | query and document *independently* | Yes — document vectors are computed once and indexed | First-stage retrieval over a large corpus |
| **Cross-encoder** | the query and document *jointly*, concatenated as one input | No — every pair is a forward pass | Reranking a shortlist |
| **Late interaction** | document tokens encoded once; the query is scored against those token vectors | The document side, yes | Between the two (ColBERT) |

DPR (Karpukhin et al., 2020) is the bi-encoder landmark: embed query and
passage separately, compare with a dot product, and search becomes an index
lookup. ColBERT (Khattab & Zaharia, 2020) keeps per-token document vectors and
scores them against the query at search time, so its late interaction sits
between the two regimes rather than in either one. A cross-encoder is the
accuracy end and the only one of the three that cannot pre-compute the document
side.

The rule of thumb follows from that table: **bi-encoder for the index,
cross-encoder for the shortlist.** Retrieve a few hundred candidates cheaply,
then rerank them expensively. If your corpus is small enough that a forward pass
per candidate fits the latency budget, you may not need a pre-computed index at
all — but say so explicitly, because the choice changes every number you
measure.

## Match the objective to the task

An embedding model is defined by its training signal, and the signal decides
what "similar" means. Match the objective to the comparison your task actually
makes.

| Your task | The comparison it makes | Start from |
| --- | --- | --- |
| Retrieval / RAG | short query vs long passage — **asymmetric** | A bi-encoder trained on query–passage pairs (DPR, 2020); add a cross-encoder reranker if shortlist accuracy binds |
| Clustering / deduplication | text vs text — **symmetric** | A sentence embedder trained for symmetric similarity (SBERT, 2019; SimCSE, 2021) |
| Semantic textual similarity | sentence vs sentence — symmetric, graded | The same sentence embedders; STS is the task class they are tuned and reported on |
| Classification | text vs fixed classifier weights | One vector per text as features; the pooling rule matters more here than model size |

Sentence-BERT (Reimers & Gurevych, 2019) is the recipe that made sentence-level
comparison practical: a siamese/triplet network is fine-tuned so that the pooled
output *is* the sentence vector, and the authors report orders-of-magnitude
speedup over running every sentence pair through BERT jointly — the two regimes
of the previous section, priced. SimCSE (Gao et al., 2021) is the contrastive
landmark on top: pass the same sentence twice through the encoder with different
dropout masks, treat the two outputs as a positive pair, and the authors report
that this label-free objective improves sentence embeddings on STS.

The mismatch to avoid: a sentence embedder tuned on symmetric STS pairs is **not
automatically a good retriever**. Retrieval is asymmetric — a short query
against a long passage — and a model trained for symmetric similarity has never
seen that shape. Pick the objective that matches the comparison, or measure both
on your own data.

## Pick a starting family, and note its year

Four eras, each a different kind of artifact. The year tells you what a model was
built to do; the reference's comparison table gives the full row, its unit, and
its training signal.

- **Contextual encoder, representation level — ELMo (2018), BERT (2019),
  RoBERTa (2019).** One vector per token occurrence, so you pool them yourself
  and the contract above is entirely your choice. ELMo (Peters et al., 2018)
  computes those vectors with a biLSTM; BERT (Devlin et al., 2019) with a
  Transformer encoder. RoBERTa (Liu et al., 2019) reported that BERT was
  undertrained and improved on it with more data and longer training, which is
  why "BERT" in a comparison table is really a *BERT-family* row: the
  tokenization and pooling details belong to the checkpoint, not the name.

- **Sentence embedder — SBERT (2019), SimCSE (2021).** Trained so that one
  pooled vector carries a whole sentence. Reach here when the comparison unit is
  a sentence or a short passage and you want cosine to be the entire scoring
  function.

- **Instruction-tuned embedder — E5 (2022), BGE (2023), Instructor (2022).** An
  instruction or a role prefix is part of the *input*, not documentation. E5
  (Wang et al., 2022) trains with weakly supervised contrastive pairs and expects
  query/passage prefixes; Instructor (Su et al., 2022) prepends a task
  instruction; BGE (Xiao et al., 2023) ships a family and training recipe in the
  same lineage. The consequence is operational: **the prefix used at indexing
  must be the prefix used at query time**, or the query and document vectors
  live in different regions of the space.

- **Truncatable — Matryoshka (2022).** Kusupati et al. (2022) train nested
  representations in which a prefix of the vector is itself a usable embedding,
  so one model serves several dimension budgets. Reach here when storage or
  latency binds (next two sections).

If none of these distinctions moves your measurement, you are not choosing
between families; you are choosing between checkpoints inside one, and the
decision table is the faster path.

## Match the model to language, domain, and length

Three constraints sit outside the task and the objective, and each can veto a
model that looks best on paper.

- **Domain shift.** BEIR (Thakur et al., 2021) evaluated retrieval models
  zero-shot across heterogeneous datasets and reported that dense retrievers
  which led in-domain often lost that advantage out of domain, with BM25
  remaining a strong zero-shot baseline. That is a *zero-shot* finding: it says
  benchmark leadership does not transfer, not that dense retrieval cannot win.
  Measure on your corpus, and benchmark BM25 before concluding a dense model
  underperforms.

- **Language.** The encoder must have been trained on your language or on a
  multilingual corpus that covers it; instruction-tuned embedders are commonly
  English-centric. Cost also depends on the tokenizer: byte-pair-style subword
  tokenization is the layer underneath every contextual model (see
  [fastText and BPE](../reference/word2vec-and-glove.md#fasttext-and-bpe)), so
  the number of tokens a document becomes — and therefore the encoder cost per
  document — is a property of the tokenizer as well as the model.

- **Length.** An embedding is one vector per input, so a long document must be
  chunked before it is embedded, and retrieval operates on passages — the unit
  DPR was trained on. Chunking strategy, index maintenance, and reranker
  orchestration are retrieval engineering and out of scope here; they are named
  only as the next decision after this one.

One more criterion belongs in the list even though this topic makes no bias
claim of its own: contextual models do not remove the social bias the static
vectors carried, so a model choice is also a bias choice. That problem has its
own page — [debiasing word embeddings](../explanation/debiasing-word-embeddings.md)
— and the honest position here is only that the representation change does not
settle it.

## Count the dimension and the cost

Dimension is a storage and latency term, not a quality knob, and it is easy to
budget before you commit:

- **Storage** is `N × d × 4` bytes for float32 vectors — a million passages at
  `d = 768` is roughly 3.1 GB before any index overhead. At `d = 1536` it is
  twice that.
- **Query-time scoring** is `O(d)` per candidate compared, so dimension
  multiplies the comparison cost as well as the memory.

This is where a Matryoshka-trained model earns its row: Kusupati et al. (2022)
report that truncating a nested representation to a prefix keeps most of its
retrieval quality at substantially reduced dimension, so one model can be
embedded at one width and served at another. Two cautions. Truncation is only
supported by models trained for it — cutting the dimension of an ordinary
embedder is just discarding information. And truncation **changes the vectors**,
so if you truncate, re-run the measurement below rather than assuming the
ranking survived.

## Measure before you commit

MTEB (Muennighoff et al., 2023) is the benchmark that made embedding models
comparable, and its own result is the warning you need: a single average hides a
wide per-task spread. Its taxonomy covers eight task types — retrieval,
reranking, clustering, classification, STS, pair classification, summarization,
and bitext mining — and a model that leads one category can sit well down
another. The average is a summary of *those* tasks, not a prediction about
yours.

So build a small evaluation of your own before you commit to a model:

1. **Collect a few hundred labelled examples in the shape your task uses** —
   query/passage pairs for retrieval, duplicate pairs for deduplication, grouped
   texts for clustering, labelled texts for classification.
2. **Score with the metric the task uses** — recall or nDCG at *k* for
   retrieval, agreement with the labels for clustering or classification,
   correlation with human similarity for STS.
3. **Compare full contracts, not model names** — same pooling, same
   normalisation, same instruction policy on both sides of every comparison.
4. **Hold out** a slice, and include the cheapest baseline that could work:
   BM25 for retrieval (Thakur et al., 2021), the static-vector pipeline you
   already have for a similarity task.
5. **Record the winner as a contract** — model, pooling, instruction,
   normalisation — so the next engineer can reproduce the vectors rather than
   re-derive them.

## Decision table

Read it as: the task and constraint you have, the default it points to, and the
thing that would send you elsewhere. Regime facts are architectural; family
results are the cited, dated claims above.

| If your task is | Default to | Move on when |
| --- | --- | --- |
| Large-corpus retrieval or RAG | A **bi-encoder** retriever (DPR-style), indexed offline | Shortlist precision binds — add a **cross-encoder** reranker over the top *k* |
| Retrieval on a domain you have not measured | **BM25** first, then a dense bi-encoder on your own data | Dense wins in-domain on your evaluation, not only on a benchmark |
| Reranking an existing shortlist | A **cross-encoder** | Latency forbids a forward pass per candidate — stay with the bi-encoder |
| Clustering or deduplication | A **sentence embedder** on a symmetric objective (SBERT / SimCSE) | Texts are long — chunk into sentences or passages first |
| Classification | A contextual encoder with **mean or CLS pooling**, one vector per text | The pooling choice measurably changes accuracy — measure it |
| Tight storage or latency | A **Matryoshka-trained** embedder, served at a truncated width | Truncation costs more quality than the budget saves |
| Instruction-following retrieval | An **instruction-tuned** embedder (E5 / Instructor) | You cannot apply the same prefix at index and query time |
| A checkpoint or recipe you must reproduce | **Whatever it used** | Never, until you have re-validated a swap on your own evaluation |

The last row is the quiet one: if you are extending a system whose vectors are
already indexed, the model is chosen for you, and this whole guide is a plan for
your *next* index, not this one.

## Diagnose an embedding model that underperforms

| Symptom | First thing to check |
| --- | --- |
| Relevant documents rank below irrelevant ones | The instruction prefix. Indexing without the passage prefix the model expects degrades retrieval silently — apply the same prefix on both sides. |
| Similarity scores all cluster near 1, or rank by length | Normalisation. Are both sides L2-normalised, and is the comparison actually cosine? |
| Two "BERT embeddings" are not comparable | Pooling or normalisation differs. Compare full contracts, not model names. |
| Strong in-domain, weak on a new corpus | The zero-shot finding behind BEIR. Re-measure on the new domain, and benchmark BM25 before blaming the model. |
| Clustering mixes texts that retrieval would separate | The objective is asymmetric (retrieval-trained) while the task is symmetric. Switch to a sentence embedder. |
| Long documents lose detail | One vector per input. Chunk into passages before embedding. |
| Quality drops after cutting the dimension | You truncated a model not trained for it, or past what its nesting supports. Re-measure; do not assume. |
| The MTEB-best model loses to a simpler one | The average does not transfer. Your own evaluation is the decision, not the leaderboard. |

## Key Takeaways

1. This is a model-consumption decision: which pretrained encoder to use,
   configured how, measured against what — not an architecture choice. Attention
   versus state-space lives in [choose a sequence model](choose-a-sequence-model.md);
   GRU versus LSTM lives in [choose a gated cell](choose-a-gated-cell.md).
2. Pin the four-part contract before comparing anything — encoder, pooling,
   instruction prefix, normalisation. Two "BERT embeddings" with different
   pooling or normalisation are incompatible, so report the contract with every
   result.
3. Decide the regime before the model: **bi-encoder for the index,
   cross-encoder for the shortlist.** The bi-encoder pre-computes document
   vectors and is approximate; the cross-encoder scores pairs jointly and cannot
   pre-compute; ColBERT's late interaction sits between them. Every retrieval
   claim names its regime.
4. Match the training objective to the comparison the task makes. Retrieval is
   asymmetric (query vs passage, DPR); clustering, deduplication, and STS are
   symmetric (SBERT, SimCSE). A sentence embedder tuned on STS is not
   automatically a good retriever.
5. Date the family you start from: representation-level contextual encoders
   (ELMo 2018, BERT 2019, RoBERTa 2019), sentence embedders (SBERT 2019,
   SimCSE 2021), instruction-tuned embedders (E5 2022, Instructor 2022, BGE
   2023), and truncatable embeddings (Matryoshka 2022). If the family does not
   move your measurement, you are choosing between checkpoints.
6. Domain, language, and length can veto a model that leads a benchmark. BEIR
   (2021) reported that in-domain dense leaders often lose their edge out of
   domain, with BM25 a strong zero-shot baseline; long documents must be chunked
   before embedding.
7. A benchmark average does not transfer. MTEB (2023) covers eight task types
   and reports a wide per-task spread; build a few-hundred-example evaluation in
   your task's shape, compare full contracts, and record the winner as one.

## References

- **Peters, M. et al. (2018). Deep contextualized word representations. NAACL.**  
  [https://arxiv.org/abs/1802.05365](https://arxiv.org/abs/1802.05365) – ELMo; contextual vectors from a biLSTM.
- **Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL.**  
  [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805) – Contextual embeddings from a Transformer encoder.
- **Liu, Y. et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach.**  
  [https://arxiv.org/abs/1907.11692](https://arxiv.org/abs/1907.11692) – The "BERT was undertrained" result behind the BERT-family row.
- **Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP.**  
  [https://arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084) – The siamese/triplet recipe for sentence vectors.
- **Gao, T., Yao, X. & Chen, D. (2021). SimCSE: Simple Contrastive Learning of Sentence Embeddings. EMNLP.**  
  [https://arxiv.org/abs/2104.08821](https://arxiv.org/abs/2104.08821) – Contrastive sentence embeddings with dropout positives.
- **Karpukhin, V. et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. EMNLP.**  
  [https://arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906) – The bi-encoder retriever.
- **Khattab, O. & Zaharia, M. (2020). ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. SIGIR.**  
  [https://arxiv.org/abs/2004.12832](https://arxiv.org/abs/2004.12832) – Late interaction between bi-encoder and cross-encoder.
- **Thakur, N. et al. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. NeurIPS Datasets and Benchmarks.**  
  [https://arxiv.org/abs/2104.08663](https://arxiv.org/abs/2104.08663) – Zero-shot retrieval and why dense retrieval is brittle out of domain.
- **Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.**  
  [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401) – The generation pattern built on a retrieval index.
- **Wang, L. et al. (2022). Text Embeddings by Weakly-Supervised Contrastive Pre-training.**  
  [https://arxiv.org/abs/2212.03533](https://arxiv.org/abs/2212.03533) – E5; instruction-prefixed embeddings.
- **Xiao, S. et al. (2024). C-Pack: Packed Resources For General Chinese Embeddings.**  
  [https://arxiv.org/abs/2309.07597](https://arxiv.org/abs/2309.07597) – BGE; the model family and its training recipe.
- **Su, H. et al. (2022). One Embedder, Any Task: Instruction-Finetuned Text Embeddings.**  
  [https://arxiv.org/abs/2212.09741](https://arxiv.org/abs/2212.09741) – Instructor; task instructions as part of the input.
- **Kusupati, A. et al. (2022). Matryoshka Representation Learning. NeurIPS.**  
  [https://arxiv.org/abs/2205.13147](https://arxiv.org/abs/2205.13147) – Nested, truncatable embeddings.
- **Muennighoff, N., Tazi, N., Magne, L. & Reimers, N. (2023). MTEB: Massive Text Embedding Benchmark. EACL.**  
  [https://arxiv.org/abs/2210.07316](https://arxiv.org/abs/2210.07316) – The task taxonomy and the per-task spread behind the average.

## Crosswalk

| Reader wants to | Go to |
| --- | --- |
| Learn it step by step | not written for this topic — an explanation-first overlay; the hands-on lessons are the static-embedding notebooks ([04](../tutorials/04-word-embeddings.ipynb), [05](../tutorials/05-word2vec-and-glove.ipynb), [06](../tutorials/06-debiasing-word-embeddings.ipynb)) |
| Do it in a project | Choose an embedding model — *this page* |
| Look up a model family, a pooling rule, or a benchmark | [Contextual embeddings](../reference/contextual-embeddings.md) |
| Understand why it works | [Contextual embeddings](../explanation/contextual-embeddings.md) |
| Learn by running it | not written for this topic — no implementation track, and pretrained models are not fixtures |

**Sibling topics (the Word Embeddings family):** the head
[`word-embeddings`](../reference/word-embeddings.md) (what a word vector _is_),
[`word2vec-and-glove`](../reference/word2vec-and-glove.md) (how vectors are
_learned_), and [`debiasing-word-embeddings`](../explanation/debiasing-word-embeddings.md)
(what bias the geometry carries). The sequence-architecture overlay is
[`modern-sequence-models`](../explanation/modern-sequence-models.md).
