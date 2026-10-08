# Word embeddings

A one-hot vector *names* a word; an embedding *describes* one. This page explains
why that difference is the whole game — why representing words as dense vectors of
learned features, instead of as indices into a vocabulary, is what lets a model
treat *king* and *queen* as similar, solve analogies by vector arithmetic, and
carry knowledge learned on a billion-word corpus into a task that has only a
handful of labels. It argues for the mechanism rather than tabulating it; for the
equations, shapes, and exact rules in lookup form, and for the hands-on version,
see the sibling views through the [Crosswalk](#crosswalk).

It assumes comfort with vectors, dot products, and the matrix–vector product, and
it treats the word vectors as *given* — a small, hand-built toy matrix whose every
answer is exact and checkable by eye — because *how* the vectors are learned
(Word2Vec, GloVe, and the rest) is the subject of a separate topic. The notation
follows Andrew Ng's Deep Learning Specialization, Course 5, Week 2 (Ng, 2018): the
embedding matrix $E$ has shape $(d \times V)$ — $d$ features down the rows, $V$
vocabulary words across the columns, one column per word.

## Contents

- [The problem with one-hot vectors](#the-problem-with-one-hot-vectors)
- [From sparse symbols to dense features](#from-sparse-symbols-to-dense-features)
- [The embedding matrix is a linear layer](#the-embedding-matrix-is-a-linear-layer)
- [What the geometry gives you](#what-the-geometry-gives-you)
- [Projecting the space to two dimensions](#projecting-the-space-to-two-dimensions)
- [Why pretrained vectors are worth reusing](#why-pretrained-vectors-are-worth-reusing)
- [The limit that motivates sequence models](#the-limit-that-motivates-sequence-models)
- [Where embeddings go from here](#where-embeddings-go-from-here)
- [Implementation Examples](#implementation-examples)
- [Crosswalk](#crosswalk)
- [References](#references)
- [Key Takeaways](#key-takeaways)

## The problem with one-hot vectors

The obvious way to hand a word to a network is the **one-hot vector**: fix an
ordering of the vocabulary, and encode the word at index $i$ as the $V$-dimensional
vector $o_w \in \{0,1\}^V$ that is $1$ at position $i$ and $0$ everywhere else. It
is unambiguous and trivial to build, and for a while it was simply how text entered
a model.

Its first cost is size — one coordinate per vocabulary word, so a realistic
vocabulary makes every input a vector of tens or hundreds of thousands of mostly-zero
entries. But sparsity is not the real problem; the real problem is **geometry**. Any
two *distinct* one-hot vectors share no non-zero coordinate, so their dot product is
exactly zero: every pair of different words is orthogonal, hence has cosine
similarity $0$, and every pair sits at the same Euclidean distance $\sqrt{2}$ from
every other. In one-hot space *cat* is exactly as close to *dog* as it is to *the*,
and *queen* is exactly as close to *king* as to *banana*. The encoding captures a
word's identity and discards every relationship between words.

That is a disaster for generalisation. A model built on one-hot inputs has to learn
how each word behaves independently, because nothing in the representation tells it
that two words are alike; what it learns about *dog* transfers nothing to *cat*. The
way out rests on an old idea in linguistics — the **distributional hypothesis**,
that a word's meaning is its pattern of use, "you shall know a word by the company
it keeps" (Harris, 1954; Firth, 1957). If words that appear in similar contexts mean
similar things, then a representation that places such words near each other would
let the model generalise across them. One-hot cannot; a dense, learned representation
can.

## From sparse symbols to dense features

The fix is to replace the $V$-dimensional sparse indicator with a dense,
low-dimensional vector of **features**, with $d \ll V$ — typically a few hundred
dimensions rather than tens of thousands. Each word gets a point in $\mathbb{R}^d$,
and because the coordinates are shared across all words, similarity becomes a matter
of degree rather than all-or-nothing (Bengio, Ducharme, Vincent & Jauvin, 2003).
This is a **featurized** representation: the vector describes the word along learned
axes instead of merely pointing at its slot in a list.

To make every claim on this page exact and checkable, the implementations work on a
planted toy space — six words in $d = 4$ dimensions, hand-written so the numbers
come out clean:

| Word | royalty (coord 0) | gender (coord 1) | coords 2–3 |
| --- | --- | --- | --- |
| `king` | $2$ | $+1$ | $0$ |
| `man` | $0$ | $+1$ | $0$ |
| `prince` | $1$ | $+1$ | $0$ |
| `princess` | $1$ | $-1$ | $0$ |
| `queen` | $2$ | $-1$ | $0$ |
| `woman` | $0$ | $-1$ | $0$ |

Here coordinate 0 is a *royalty* score and coordinate 1 is a *gender* sign ($+1$
masculine, $-1$ feminine); the last two coordinates are zero, so the data actually
lives in a 2-D plane inside $\mathbb{R}^4$. This hand-labelling is a teaching device,
and it is worth being blunt about what is real and what is not. In an embedding
learned from a corpus you **cannot** point at coordinate 37 and call it "royalty" —
the individual axes carry no interpretation at all. What survives into real
embeddings, and what this page is really about, is that *directions and offsets* in
the space are meaningful even when the axes are not. The toy simply makes two of
those directions line up with the coordinate axes so the arithmetic is legible.

One boundary, stated once. The "gender" direction here exists only to make the
analogy geometry concrete. Whether such a direction *should* exist in a real
embedding, and what social harm it encodes when it does, is a serious question — and
it is the subject of a separate topic, not an argument this page makes from the
example.

## The embedding matrix is a linear layer

Stack the word vectors as the columns of one matrix and you have the **embedding
matrix** $E$ of shape $(d \times V)$. Retrieving word $i$'s vector is then a matrix
product with its one-hot column:

$$e_w \;=\; E\,o_w \;=\; E_{:,\,i} \;\in\; \mathbb{R}^{d}.$$

This identity is the spine of the whole topic. Multiplying $E$ by a one-hot vector
selects the one column whose coordinate is $1$ and zeros out the rest, so $E\,o_w$ is
*exactly* the $i$-th column of $E$. An **embedding layer is therefore a bias-free
linear layer whose input happens to be one-hot** — nothing more exotic. The practical
consequence is that you never actually form the product: it is $V-1$ multiply-adds
that are all multiplications by zero. You **gather** column $i$ by its index, which is
why every framework calls this a *lookup table*. The identity tells you the lookup is
legitimate; the index tells you how to compute it cheaply.

Reading the layer this way clears up two things that otherwise confuse people. First,
the embedding matrix *is* the parameters — training the lookup table is training the
word vectors, with the one-hot input supplying the gradient to exactly the row that
was looked up. Second, because the layer is just a linear map, its matrix can be
**tied** to another linear layer elsewhere in a model — for example the output
projection of a language model — so one set of weights serves as both the input
embedding and the output classifier.

A convention worth pinning down, because mixing it up transposes your whole matrix.
This page uses Ng's $(d \times V)$ layout, word-per-column. PyTorch's
`torch.nn.Embedding` stores its weight the other way, as $(V \times d)$ —
`num_embeddings` rows by `embedding_dim` columns — so there a word vector is a *row*,
`weight[i, :]`, and `weight` equals $E^{\top}$. Neither is more correct; they are
transposes of each other, and the PyTorch idiom track documents the translation so no
one silently flip $d$ and $V$.

## What the geometry gives you

Once words are points in $\mathbb{R}^d$, questions about meaning become questions
about geometry, and you can answer them by reading the vectors rather than training
anything further.

**Cosine similarity** measures the angle between two vectors, ignoring their lengths:

$$\cos(u, v) \;=\; \frac{u \cdot v}{\lVert u\rVert_2\,\lVert v\rVert_2}.$$

It runs from $+1$ for same-direction vectors through $0$ for orthogonal ones to $-1$
for opposite ones. Direction rather than distance is the right currency here because
a vector's *length* in an embedding space tends to track nuisance factors like raw
word frequency, while its *direction* carries the semantics. On the toy space the
numbers are exact and tell a story: $\cos(\textit{king}, \textit{queen}) = \tfrac{3}{5}
= 0.6$, whereas $\cos(\textit{man}, \textit{woman}) = -1$ — *man* and *woman* share the
same (zero) royalty and sit at opposite gender signs, so they point in exactly
opposite directions. And $\cos(\textit{king}, \textit{man}) = 1/\sqrt{5} \approx 0.447$,
*smaller* than $\cos(\textit{king}, \textit{queen})$: a king is geometrically closer to
a queen than to a man, because royalty dominates gender in this space. One-hot could
never express that ordering; every one of those cosines would have been $0$.

**Nearest neighbours** fall straight out: rank every other word by cosine to the
query, and read off the top of the list. The ranking is made deterministic by
excluding the query itself and breaking ties by ascending index, so on the toy space
it is a fixed table — `king`→`prince`, `queen`→`princess`, `prince`→`king`,
`princess`→`queen`, and both `man` and `woman`→`prince`/`princess` respectively. No
training produced this; it was latent in the vectors.

The property that made embeddings famous is the **analogy**, or *parallelogram*,
structure (Mikolov, Yih & Zweig, 2013): relationships between words show up as
roughly constant vector *offsets*. The canonical example is

$$\textit{king} - \textit{man} + \textit{woman} \;\approx\; \textit{queen}.$$

Geometrically, the displacement from *man* to *king* — "add royalty" — is about the
same displacement as from *woman* to *queen*, so subtracting *man* and adding *woman*
to *king* lands you on *queen*. The standard way to solve "$a$ is to $b$ as $c$ is to
?" is **3CosAdd**: form the target

$$t \;=\; e_b - e_a + e_c,$$

and return the word whose embedding is most cosine-similar to $t$, **excluding $a$,
$b$, and $c$ from the candidates**. That exclusion is not a detail — the offsets are
small next to the words themselves, so the single vector closest to $t$ is usually one
of the input words, and without the exclusion $\textit{king} - \textit{man} +
\textit{woman}$ simply returns *king*. On the toy space, taking $a = \textit{man}$,
$b = \textit{king}$, $c = \textit{woman}$ gives $t = [2, -1, 0, 0]$, whose nearest word
(after removing *man*, *king*, and *woman*) is *queen* at cosine $1.0$, ahead of the
runner-up *princess* at $3/\sqrt{10} \approx 0.949$. The additive scorer is not the only
one; **3CosMul** (Levy, Goldberg & Dagan, 2015) multiplies the similarities instead of
adding them, which keeps any single large term from dominating and is more robust on
real corpora — it is named here but not needed on a space this clean.

Why should linear arithmetic recover a relationship at all? Because good embedding
spaces have genuine **linear substructure**: the training objectives behind Word2Vec
and, explicitly, GloVe arrange the geometry so that ratios of co-occurrence
probabilities correspond to vector *differences*, which is exactly what turns an
analogy into a subtraction and an addition (Pennington, Socher & Manning, 2014;
Mikolov, Yih & Zweig, 2013). This page relies on that structure; the sibling
word2vec-and-GloVe topic derives where it comes from.

## Projecting the space to two dimensions

A few hundred dimensions is unviewable, so to *inspect* an embedding space we project
it down to two. **Principal component analysis** is the honest first tool: it finds the
linear projection that preserves the most variance. Done from scratch, it is a singular
value decomposition. Lay the words out as rows, $X = E^{\top}$ of shape $(V \times d)$,
subtract the mean word vector to centre them, $X_c = X - \bar{x}$, take the SVD
$X_c = U\Sigma W^{\top}$, and keep the first two rows of $W^{\top}$ — the directions of
greatest variance. The 2-D coordinates are the centred data projected onto them,
$X_c\,W_{:,\,:2}$. One wrinkle: an SVD fixes each principal direction only up to sign,
so two runs could return mirror-image pictures; pinning a convention — flip each
component so its largest-magnitude entry is positive — makes the plot reproducible
without changing anything it means.

The toy space is **rank-2 by construction** (only royalty and gender carry signal; the
other two coordinates are zero), so projecting it to two dimensions loses *nothing* —
PCA recovers the semantic plane exactly, with reconstruction error $0$ and every
pairwise distance preserved. After centring, the gender axis carries more variance
(sum of squares $6$) than royalty ($4$), so the first principal component lines up with
gender and the second with royalty. That exactness is a property of the planted data,
not of PCA in general.

And it names PCA's limit. A linear projection can only reveal structure that survives
being flattened linearly; the crowded, cluster-revealing scatter plots people actually
publish for real embeddings come from **t-SNE** (van der Maaten & Hinton, 2008), a
nonlinear method that preserves local neighbourhoods rather than global variance. The
trade is that t-SNE is stochastic, depends on hyperparameters, and distorts
between-cluster distances, so its pictures are suggestive rather than quantitative. PCA
is the trustworthy first look; t-SNE is the prettier second one, and implementing it is
its own topic.

## Why pretrained vectors are worth reusing

The property that put embeddings into nearly every NLP pipeline for a decade is
**transfer**. A few hundred dimensions, trained once on a very large unlabelled corpus,
capture broad regularities of a language — which words behave alike, which relationships
are systematic. You can then lift that matrix into a downstream task that has only a few
thousand labelled examples, far too few to learn good word representations from scratch,
and the task inherits the corpus's knowledge for free (Ng, 2018; Bengio et al., 2003).
The downstream model starts already knowing that *excellent* and *superb* are close, so
it can generalise from one to the other even if only one appeared in its tiny training
set. For years this is why the first layer of most NLP models was simply a pretrained
embedding matrix — Word2Vec or GloVe vectors — sometimes frozen, sometimes fine-tuned. As
of 2026 that specific recipe has largely given way to contextual encoders, for reasons
the final section takes up, but the transfer argument is the same one those successors
inherited.

## The limit that motivates sequence models

The quickest way to turn per-word vectors into a per-sentence vector is to **average**
them — a bag-of-embeddings. It is cheap and a surprisingly strong baseline for coarse
tasks like topic or sentiment classification, because the average lands in the region of
the space the sentence's words occupy. But it has a structural blind spot: averaging is
**permutation-invariant**. Any two phrases built from the same multiset of words collapse
to the identical vector, so "not good" and "good not" are indistinguishable, and — the
point that matters — the average has no mechanism to let *not* modify *good*. The averaged
vector of "not good" sits near the average of its words, which is roughly where "good"
already is, so a bag-of-embeddings classifier rates "not good" about as positive as
"good". Averaging discards word order, and word order is where negation, scope, and much
of syntax live.

That limit is not a flaw to patch; it is the precise reason a different model exists. To
let *not* flip what follows it, a model has to read the words **in order** and compose
them as it goes, which is exactly what a recurrent network does — the order-aware version
of this same sentiment task is the many-to-one wiring of the
[recurrent neural networks](recurrent-neural-networks.md) topic. The order-blindness of
averaging is demonstrated directly in this topic's notebook; here it is the hinge that
connects static word vectors to the sequence models that consume them.

## Where embeddings go from here

This page deliberately stopped at what a word vector *is* and what you can read off it.
Three threads continue, each its own topic.

**How the vectors are learned.** We took them as given. Word2Vec's skip-gram and CBOW
objectives (Mikolov, Chen, Corrado & Dean, 2013) and GloVe's weighted least squares over
co-occurrence counts (Pennington, Socher & Manning, 2014) are the training procedures that
actually produce embeddings with the geometry this page relied on — including the linear
substructure that makes analogies work. The word2vec-and-GloVe topic derives them from
scratch.

**Bias.** The same geometry that yields clean analogies also absorbs the stereotypes
present in the training text. The very kind of gender direction that makes
$\textit{king} - \textit{man} + \textit{woman} = \textit{queen}$ work is the one Bolukbasi,
Chang, Zou, Saligrama & Kalai (2016) found aligning *programmer* with men and *homemaker*
with women — and Gonen & Goldberg (2019) showed that geometric "debiasing" tends to hide
such bias rather than remove it.

**Static to contextual.** A static embedding assigns one vector per word *type*, so the
river *bank* and the money *bank* are forced to share a vector — the ceiling of the whole
approach. As of 2026 that ceiling has largely been lifted by **contextual** embeddings,
which give a word a different vector in every sentence: ELMo (Peters et al., 2018) and
BERT (Devlin et al., 2019) read the surrounding words, and sentence-level models such as
SBERT (Reimers & Gurevych, 2019), compared on benchmarks like MTEB (Muennighoff, Tazi,
Magne & Reimers, 2023), now power search, retrieval, and retrieval-augmented generation.
The contextual-embeddings topic surveys that overlay.

## Implementation Examples

### Complete Implementations:

> #### **[Word embeddings (NumPy)](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/embeddings/word_embeddings.py)** - The from-scratch teaching reference behind every number on this page: the planted toy matrix, the lookup identity `E o_w = E[:, i]`, cosine similarity with a zero-vector guard, deterministic nearest neighbours, the 3CosAdd analogy with the input words excluded, and a from-scratch PCA-via-SVD projection to 2-D with a sign convention that makes the picture reproducible.
> #### **[Word embeddings (PyTorch idiom port)](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/pytorch/embeddings/word_embeddings.py)** - The same vectors in a `torch.nn.Embedding`, the way the lookup is really written in PyTorch: a gather by index rather than the one-hot product, over a weight stored as `(V, d)` — one word per *row*, so `weight == E.T`. An idiom track, not a parity port — it shows the transpose a reader must get right, the `from_pretrained(freeze=True)` default (the opposite of the trainable `nn.Embedding(V, d)` constructor), and weight tying as the forward pointer to the language-modeling topic. Checked on its own correctness: the gather equals the NumPy `lookup` exactly for every word.

## Crosswalk

| Reader wants to | Go to |
| --- | --- |
| Learn it step by step | [Word embeddings (notebook)](../tutorials/04-word-embeddings.ipynb) |
| Look up an equation or rule | [Word embeddings (reference)](../reference/word-embeddings.md) |
| Understand why it works | Word embeddings — *this page* |
| Learn by running it | [Word embeddings (notebook)](../tutorials/04-word-embeddings.ipynb) |

## References

- **Harris, Z. (1954). Distributional Structure. _Word_ 10(2–3), 146–162.** – The distributional hypothesis: a word's meaning is its pattern of use in context.
- **Firth, J. R. (1957). A synopsis of linguistic theory 1930–1955. In _Studies in Linguistic Analysis_. Blackwell.** – "You shall know a word by the company it keeps."
- **Bengio, Y., Ducharme, R., Vincent, P. & Jauvin, C. (2003). A Neural Probabilistic Language Model. _Journal of Machine Learning Research_ 3, 1137–1155.** – Learned distributed word-feature vectors, and the transfer argument for them.
- **Mikolov, T., Chen, K., Corrado, G. & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space.**  
  [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781) – Word2Vec: the skip-gram and CBOW training objectives.
- **Mikolov, T., Yih, W. & Zweig, G. (2013). Linguistic Regularities in Continuous Space Word Representations. NAACL-HLT, 746–751.** – The analogy / parallelogram property this page explains.
- **Pennington, J., Socher, R. & Manning, C. (2014). GloVe: Global Vectors for Word Representation. EMNLP.**  
  [https://nlp.stanford.edu/pubs/glove.pdf](https://nlp.stanford.edu/pubs/glove.pdf) – The linear substructure that makes analogies linear.
- **Levy, O., Goldberg, Y. & Dagan, I. (2015). Improving Distributional Similarity with Lessons Learned from Word Embeddings. _TACL_ 3, 211–225.** – 3CosMul and the hyperparameters that matter for analogy recovery.
- **van der Maaten, L. & Hinton, G. (2008). Visualizing Data using t-SNE. _Journal of Machine Learning Research_ 9, 2579–2605.** – The nonlinear projection contrasted here with PCA.
- **Bolukbasi, T., Chang, K.-W., Zou, J., Saligrama, V. & Kalai, A. (2016). Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings. NeurIPS.**  
  [https://arxiv.org/abs/1607.06520](https://arxiv.org/abs/1607.06520) – The gender direction and the stereotypes it carries.
- **Gonen, H. & Goldberg, Y. (2019). Lipstick on a Pig: Debiasing Methods Cover up Systematic Gender Biases in Word Embeddings. NAACL.**  
  [https://arxiv.org/abs/1903.03862](https://arxiv.org/abs/1903.03862) – Why geometric debiasing hides rather than removes bias.
- **Peters, M., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K. & Zettlemoyer, L. (2018). Deep contextualized word representations. NAACL.**  
  [https://arxiv.org/abs/1802.05365](https://arxiv.org/abs/1802.05365) – ELMo: a different vector per occurrence.
- **Devlin, J., Chang, M.-W., Lee, K. & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL.**  
  [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805) – Contextual embeddings from a Transformer encoder.
- **Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP.**  
  [https://arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084) – Sentence- and text-level embeddings.
- **Muennighoff, N., Tazi, N., Magne, L. & Reimers, N. (2023). MTEB: Massive Text Embedding Benchmark. EACL.**  
  [https://arxiv.org/abs/2210.07316](https://arxiv.org/abs/2210.07316) – How embedding models are compared today.
- **Ng, A. (2018). Deep Learning Specialization, Course 5 (Sequence Models), Week 2.**  
  [https://www.coursera.org/specializations/deep-learning](https://www.coursera.org/specializations/deep-learning) – The source of this family's framing and the $(d \times V)$ notation for $E$.

## Key Takeaways

1. One-hot vectors encode only identity: any two distinct words are orthogonal and
   equidistant, so the representation carries no similarity and a model built on it
   cannot generalise from one word to another.
2. A dense, featurized vector fixes this by placing similar words near each other;
   the coordinates of a *learned* embedding are not individually interpretable, but
   directions and offsets in the space are, which is where all the usefulness lives.
3. An embedding layer *is* a bias-free linear layer on a one-hot input — $e_w = E\,o_w
   = E_{:,\,i}$ — so in practice you gather column $i$ by index rather than forming the
   product; the weights are the word vectors. PyTorch stores $E^{\top}$, $(V \times d)$,
   so a word vector is a row.
4. Geometry becomes semantics: cosine similarity ranks by direction (on the toy space
   $\cos(\textit{king}, \textit{queen}) = 0.6$, $\cos(\textit{man}, \textit{woman}) = -1$),
   nearest neighbours fall out of that ranking, and analogies solve by 3CosAdd,
   $t = e_b - e_a + e_c$, with the three input words excluded from the candidates.
5. Analogy-as-arithmetic works because good embedding spaces have linear substructure;
   PCA via SVD gives an honest, variance-preserving 2-D view (exact on the rank-2 toy),
   while t-SNE gives the nonlinear, cluster-revealing picture at the cost of quantitative
   meaning.
6. Pretrained vectors transfer: a few hundred dimensions trained on a large corpus carry
   broad language knowledge into small-data tasks — the recipe that made embeddings
   ubiquitous.
7. Averaging word vectors into a sentence vector is order-blind — "not good" reads about
   as positive as "good" — which is precisely the limitation that motivates the
   order-aware sequence models, and the thread that continues into how embeddings are
   learned, how they encode bias, and how contextual models replaced the static ones.
