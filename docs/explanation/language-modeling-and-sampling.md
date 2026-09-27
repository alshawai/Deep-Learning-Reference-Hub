# Language modeling and sampling

A language model assigns a probability to a sequence, and — run forward — it
*generates* one. This page explains why the character-level recurrent language
model works: why decomposing the probability of a sequence into next-character
predictions turns a classifier into a generator, why the training signal is
cross-entropy and what perplexity measures, why sampling with a temperature knob
produces text that is varied without being noise, and why training the model at
all depends on gradient clipping. It argues for the mechanics rather than
tabulating them; for the equations, shapes, and defaults in lookup form, and for
the in-project task, see the sibling views through the [Crosswalk](#crosswalk).

It assumes the vanilla RNN of this family's
[recurrent neural networks](recurrent-neural-networks.md) topic — the recurrence
$a^{\langle t\rangle} = \tanh(W_{aa}\,a^{\langle t-1\rangle} + W_{ax}\,x^{\langle t\rangle} + b_a)$,
its unrolling and backpropagation through time, the exploding/vanishing-gradient
story, and the softmax cross-entropy read-out — and builds the language model on
top of that one cell. The notation is that topic's, verbatim: activations
$a^{\langle t\rangle}$, inputs $x^{\langle t\rangle}$, predictions
$\hat{y}^{\langle t\rangle}$, read-out weights $W_{ya}, b_y$, following Andrew
Ng's Deep Learning Specialization, Course 5, Week 1 (Ng, 2018).

## Contents

- [Why a language model factorises a sequence](#why-a-language-model-factorises-a-sequence)
- [Predicting the next character trains a generator](#predicting-the-next-character-trains-a-generator)
- [Cross-entropy, and why we read it as perplexity](#cross-entropy-and-why-we-read-it-as-perplexity)
- [Sampling: running the predictor as a generator](#sampling-running-the-predictor-as-a-generator)
- [Why temperature works](#why-temperature-works)
- [Why training needs gradient clipping](#why-training-needs-gradient-clipping)
- [Why there is one implementation, and why it is PyTorch](#why-there-is-one-implementation-and-why-it-is-pytorch)
- [Implementation Examples](#implementation-examples)
- [Crosswalk](#crosswalk)
- [References](#references)
- [Key Takeaways](#key-takeaways)

## Why a language model factorises a sequence

Modelling the joint probability of a whole sequence directly is hopeless: a
length-$T$ string over a vocabulary of size $V$ has $V^{T}$ possible values, so
no table and no fixed-output network could enumerate them. The way out is not an
approximation but an identity. The **chain rule of probability** rewrites *any*
joint distribution as a product of conditionals, exactly:

$$P\big(x^{\langle 1\rangle}, \ldots, x^{\langle T\rangle}\big) = \prod_{t=1}^{T} P\big(x^{\langle t\rangle} \mid x^{\langle 1\rangle}, \ldots, x^{\langle t-1\rangle}\big)$$

Nothing has been assumed away here — this is the same distribution, refactored.
What it buys is a change of problem: instead of scoring an entire sequence at
once, the model only ever has to answer one question, *given everything so far,
what comes next?*, and multiply the answers. This is the neural language model of
Bengio, Ducharme, Vincent & Jauvin (2003), realised recurrently by Mikolov et
al. (2010).

The RNN is built for exactly this question. Its hidden state $a^{\langle t\rangle}$
is a fixed-size, learned summary of everything read up to step $t$ — the lossy
running compression argued for in the RNN topic — so the read-out

$$\hat{y}^{\langle t\rangle} = \mathrm{softmax}\big(W_{ya}\,a^{\langle t\rangle} + b_y\big)$$

is a distribution over the next character, and it *is* the conditional factor
$P(x^{\langle t+1\rangle} \mid x^{\langle 1\rangle}, \ldots, x^{\langle t\rangle})$.
One network, with weights shared across every step, supplies every factor in the
product. This is where the RNN's parameter sharing pays off as *probability*: an
$n$-gram model would have to truncate the history to the last $n-1$ symbols and
still store a table that grows as $V^{n}$, whereas the recurrent state carries —
in principle — unbounded context at fixed parameter cost. Working
**character-level** shrinks the vocabulary to a few dozen symbols (the sorted
unique characters of the corpus, including the terminating newline `\n`), so the
softmax is small and the model learns spelling, word boundaries, and short
grammar from raw text with no tokeniser (Sutskever, Martens & Hinton, 2011;
Karpathy, 2015).

## Predicting the next character trains a generator

If each factor is "predict the next character", then the **training target at
step $t$ is simply the next input**, $y^{\langle t\rangle} = x^{\langle t+1\rangle}$.
The corpus is therefore its own label: feed the text in, and ask the model to
reproduce it shifted one position to the left. The first input
$x^{\langle 1\rangle}$ is the zero vector — there is no prior character to
condition on — and the last target is the newline `\n`, so the model also learns
*where a sequence ends*, which is what lets a sample stop itself.

The reason this trains a *generator*, and not merely a next-character classifier,
is that the two objectives are the same one. Cross-entropy against the true next
character is the negative log-probability the model assigns to that character, so
summing it over the sequence is exactly the negative log-likelihood of the real
text under the factorised model:

$$\sum_{t=1}^{T} -\ln \hat{y}^{\langle t\rangle}_{\,x^{\langle t+1\rangle}} = -\ln \prod_{t=1}^{T} P\big(x^{\langle t+1\rangle}\mid x^{\langle \le t\rangle}\big) = -\ln P\big(x^{\langle 1\rangle}, \ldots, x^{\langle T\rangle}\big).$$

Minimising next-character error is maximising the probability the model places on
real sequences. A model that predicts well is, by construction, a model under
which realistic text is probable — and sampling from its own predictions walks
that same distribution forward into new text. The classifier and the generator
are one object read two ways.

One asymmetry is worth naming as a cost. During training the model always
conditions on the *true* previous characters (this is *teacher forcing*); during
generation it must condition on the characters it *itself* sampled. A single
early mistake therefore lands the model in a context it never saw while training —
the *exposure bias* of autoregressive models. Character-level RNN generation
tolerates it well enough to be the classic demonstration it is (Karpathy, 2015),
but it is the seam along which longer generations drift.

## Cross-entropy, and why we read it as perplexity

Maximum likelihood, spelled out above, is minimising cross-entropy. The figure
the language model reports is the **per-character** cross-entropy — the mean
negative log-probability over every predicted token, in nats (natural log) —
rather than a per-sequence sum, because dividing by the sequence length makes the
number comparable across sequences of different length. A model that assigns
probability $\tfrac{1}{V}$ to each of $V$ characters scores $\ln V$ nats per
character no matter how long the text is.

**Perplexity** is that per-character cross-entropy, exponentiated:

$$\mathrm{PPL} = \exp(\mathrm{CE}).$$

The exponential is not decoration; it converts an abstract quantity of nats into
a count with a plain meaning. Perplexity is the model's **effective branching
factor** — the number of equally-likely next characters the model is, on average,
as uncertain among (Goodfellow, Bengio & Courville, 2016, Ch. 10). A perplexity
of $4$ means the model is as unsure as if it were choosing uniformly among four
characters at each step; the floor is $1$, a model certain of every next
character, and there is no gain in reporting nats and their exponential
separately.

This reading is what makes the **initialisation anchor** a genuine check rather
than a coincidence. Before training, the read-out weights are small and random,
the logits are near zero, and the softmax is therefore near uniform over the $V$
characters — the model *knows nothing*. Its per-character cross-entropy is then

$$\mathrm{CE} \approx -\ln\tfrac{1}{V} = \ln V, \qquad \mathrm{PPL} \approx V,$$

the "knows nothing" baseline: a fresh model is exactly as perplexed as the
alphabet is large. A freshly built model whose figure sits far from $\ln V$ has a
bug in its vocabulary size, its loss reduction, or its logarithm base long before
any question of learning arises — which is why this topic, having no from-scratch
NumPy reference to check numerically against, leans on the $\ln V$ anchor as its
correctness signal. As training proceeds the model concentrates probability on
the characters that actually follow, and the perplexity falls below $V$ toward
the true branching of the corpus.

## Sampling: running the predictor as a generator

Generation reverses the training direction. Instead of feeding the true text in
and reading the loss out, the model feeds *itself*: from the start-of-sequence
state it produces a distribution over the first character, draws one, feeds that
draw back in as the next input, and repeats. The autoregressive factorisation
demands exactly this — each conditional is over the characters *already produced*,
so the loop must condition on its own history — and the RNN's hidden state is
what carries that history forward.

The one decision that separates a generator from a decoder is that it **draws**
from the distribution rather than taking its argmax. Greedy argmax generation is
deterministic and, from a character model, usually degenerate: it falls into the
single most probable continuation and loops on it. Sampling instead honours the
whole distribution the model learned, so rare-but-plausible continuations get
their share of the mass, and the output has the variety of real text (Graves,
2013). Making the draw come from an **explicitly-passed, seeded** random generator
keeps that stochastic process reproducible — the same seed yields the same string,
which is what lets a sampled sequence be an assertion in a test.

The mechanism, framework-agnostic, is short enough to read whole:

```python
import numpy as np


def softmax(z):
    """Stable softmax over a length-V logit vector."""
    z = z - np.max(z)  # subtract the max: exp never overflows
    e = np.exp(z)
    return e / np.sum(e)


def sample(
    step, init_state, char_to_ix, ix_to_char, seed, temperature=1.0, max_length=50
):
    """Generate one sequence autoregressively from a trained char-LM.

    `step(x, state) -> (logits, next_state)` runs one recurrent cell step for
    any framework and returns a length-V logit vector; everything else here is
    the sampling mechanism itself.
    """
    rng = np.random.default_rng(seed)  # explicit, seeded RNG -> reproducible
    vocab_size = len(char_to_ix)
    newline = char_to_ix["\n"]

    x = np.zeros(vocab_size)  # x^{<1>} = 0: no prior character
    state = init_state  # a^{<0>} = 0
    drawn = []

    for _ in range(max_length):  # max_length is the hard cap L
        logits, state = step(x, state)  # read-out logits z^{<t>}
        p = softmax(logits / temperature)  # temperature-scaled distribution p^{<t>}
        idx = rng.choice(vocab_size, p=p)  # multinomial draw i^{<t>} ~ p^{<t>}
        drawn.append(idx)
        if idx == newline:  # stop as soon as "\n" is drawn ...
            break
        x = np.zeros(vocab_size)  # ... otherwise feed the draw back
        x[idx] = 1.0  # as the one-hot input x^{<t+1>}

    return "".join(ix_to_char[i] for i in drawn)
```

Two guards keep the loop honest. The draw stops as soon as the newline `\n` is
drawn — the end-of-sequence symbol the shifted targets taught — and the hard cap
`max_length` (the $L$ of the reference) guarantees termination even from an
*untrained* model, whose near-uniform distribution may place too little mass on
`\n` to ever end on its own. The canonical PyTorch module realises this same loop
with `torch.multinomial` drawing from a seeded `torch.Generator`; the code above
is the one place in this topic where the mechanism is shown as code.

## Why temperature works

The single line `softmax(logits / temperature)` is the one knob that shapes
generation, and it is worth seeing *why* dividing the logits by a scalar reshapes
the distribution. Temperature $T > 0$ rescales the logits before the softmax:

$$p^{\langle t\rangle}_i = \frac{\exp\!\big(z^{\langle t\rangle}_i / T\big)}{\sum_j \exp\!\big(z^{\langle t\rangle}_j / T\big)}.$$

Because the softmax depends only on *differences* between logits, scaling those
differences is what changes the shape:

- **$T = 1$** leaves the logits untouched: the sample is drawn from the model's
  own learned distribution.
- **$T \to 0^{+}$** blows up every logit gap, so the largest logit runs away from
  the rest and the softmax collapses onto the argmax — deterministic, greedy
  decoding as a limit of sampling.
- **$T \to \infty$** drives every $z_i / T$ toward zero, so the logits become
  equal and the distribution flattens to uniform — maximum entropy, maximum
  diversity, and no faithfulness to what the model learned.

So temperature slides monotonically between "the safest, most probable
continuation" and "anything at all", and it does so by *sharpening or flattening*
the distribution without ever reordering it — the most probable character stays
the most probable at every $T$. That is the property that makes temperature the
natural diversity control rather than, say, adding noise to the logits: it
changes only the confidence, not the model's ranking (Graves, 2013). The cost is
the obvious trade — too low and the text repeats and loops; too high and it
dissolves into gibberish. As of 2026 the same temperature knob is still the
front-line sampling control on transformer-scale language models, for the same
reason it works here.

## Why training needs gradient clipping

A character language model is trained by backpropagation through time over the
whole unrolled sequence, so it inherits the vanilla RNN's central hazard
unchanged. The gradient that reaches an early step is a product of one recurrent
Jacobian per step, and a product of many nearly-identical matrices grows or
shrinks geometrically in the number of steps (Bengio, Simard & Frasconi, 1994;
Pascanu, Mikolov & Bengio, 2013). When that product grows, a single batch can
produce an enormous gradient that throws the weights to infinity or `NaN` in one
update — the exploding-gradient failure the RNN topic derives — and long training
sequences make it a routine event, not an edge case, for a char-LM.

**Global-norm gradient clipping** is the remedy that makes this model trainable.
Every parameter gradient is stacked into one vector, its L2 norm is taken, and if
that norm exceeds a threshold $v$, *all* gradients are rescaled by the same factor
$v / \lVert g\rVert_2$:

$$g \leftarrow g \cdot \min\!\Big(1,\ \frac{v}{\lVert g\rVert_2}\Big).$$

Because one scalar multiplies the whole gradient at once, clipping caps the
*length* of the step while leaving its *direction* exactly where the gradients
pointed — a global norm, rather than clipping each parameter independently, is
what preserves that direction. It is worth being clear about what clipping does
*not* do: it does nothing for the vanishing case (a gradient that has already
decayed to zero, rescaled, is still zero), so the long-range dependencies a
vanilla char-LM struggles to learn are not rescued by it — that is the work of the
gated cells in the [LSTM and GRU](lstm-and-gru.md) topic. Clipping keeps training
from diverging; it does not extend the memory of the model. Both this topic and
the RNN topic's from-scratch `clip_gradients` implement precisely this rule, and
the canonical module reuses `torch.nn.utils.clip_grad_norm_`.

## Why there is one implementation, and why it is PyTorch

This is the hub's first topic whose canonical implementation is a framework port
rather than a from-scratch NumPy module, and the reason is a deliberate one worth
stating. The differentiable mathematics here is *not new*: the recurrent cell and
its backpropagation through time are owned by the
[recurrent neural networks](recurrent-neural-networks.md) topic and its NumPy
reference. What the language model adds on top — the shifted-target training loop,
the perplexity metric, and the sampling loop above — is *orchestration the
framework does not hide*. A from-scratch NumPy module would therefore restate
explicit Python loops that expose no mechanism the reader cannot already see in
the equations and the snippet on this page, rather than revealing hidden
gradients the way the RNN's own NumPy module does. With no new math to expose, a
second implementation would be redundant, so the topic is realised once, in
PyTorch — `torch.nn.RNN` with an `nn.Linear` read-out, `F.cross_entropy`,
`clip_grad_norm_`, and a seeded `torch.multinomial` — and that module is checked
on its own correctness against the $\ln V$ anchor rather than against a parity
reference.

## Implementation Examples

### Complete Implementations:

> #### **[Character-level language model — PyTorch (canonical)](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/pytorch/sequence/language_model.py)** - The char-RNN language model on `torch.nn.RNN` + `nn.Linear`, with per-character cross-entropy and perplexity through `F.cross_entropy`, global-norm clipping through `clip_grad_norm_`, and the seeded temperature sampling loop explained above through `torch.multinomial`. Checked on its own correctness against the shared fixture — the $\approx \ln V$ initialisation anchor, a strictly decreasing training loss, byte-for-byte sampling reproducibility, a shuffled-target discrimination test, and a `clip_grad_norm_`-versus-`clip_gradients` cross-check — because this topic has no from-scratch NumPy reference to hold it to. The docs build installs no framework, so there is no generated API page.

## Crosswalk

Live links are added as each sibling ships; plain-text entries are not yet
published.

| Reader wants to | Go to |
| --- | --- |
| Learn it step by step | [Language modeling and sampling (notebook)](../tutorials/03-language-modeling-and-sampling.ipynb) |
| Do it in a project | [Train a language model](../how-to/train-a-language-model.md) |
| Look up the factorisation or sampling algorithm | [Language modeling and sampling](../reference/language-modeling-and-sampling.md) |
| Understand why it works | Language modeling and sampling — *this page* |
| Learn by running it | [Language modeling and sampling (notebook)](../tutorials/03-language-modeling-and-sampling.ipynb) |

## References

- **Bengio, Y., Ducharme, R., Vincent, P. & Jauvin, C. (2003). A Neural Probabilistic Language Model. _Journal of Machine Learning Research_ 3, 1137–1155.** – The neural language-model factorisation this page derives.
- **Mikolov, T., Karafiát, M., Burget, L., Černocký, J. & Khudanpur, S. (2010). Recurrent neural network based language model. INTERSPEECH.** – The recurrent realisation of that factorisation.
- **Sutskever, I., Martens, J. & Hinton, G. (2011). Generating Text with Recurrent Neural Networks. ICML.** – Character-level generation from a recurrent network.
- **Graves, A. (2013). Generating Sequences With Recurrent Neural Networks.**  
  [https://arxiv.org/abs/1308.0850](https://arxiv.org/abs/1308.0850) – Sampling, sequence generation, and the temperature-scaled draw.
- **Karpathy, A. (2015). The Unreasonable Effectiveness of Recurrent Neural Networks.**  
  [https://karpathy.github.io/2015/05/21/rnn-effectiveness/](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) – The char-RNN language model and sampling, in practitioner form.
- **Bengio, Y., Simard, P. & Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult. _IEEE Transactions on Neural Networks_ 5(2), 157–166.** – The original vanishing/exploding-gradient analysis this page's clipping section rests on.
- **Pascanu, R., Mikolov, T. & Bengio, Y. (2013). On the difficulty of training Recurrent Neural Networks. ICML.**  
  [https://arxiv.org/abs/1211.5063](https://arxiv.org/abs/1211.5063) – Exploding gradients and global-norm clipping.
- **Goodfellow, I., Bengio, Y. & Courville, A. (2016). _Deep Learning_, Ch. 10.**  
  [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/) – Cross-entropy and perplexity as the language-model training signal.
- **Ng, A. (2018). Deep Learning Specialization, Course 5 (Sequence Models), Week 1.**  
  [https://www.coursera.org/specializations/deep-learning](https://www.coursera.org/specializations/deep-learning) – The source of this family's notation, and the _Dinosaur Island_ character-level assignment this topic realises.

## Key Takeaways

1. The chain rule of probability factorises a sequence *exactly* into a product
   of next-character conditionals, $P(x^{\langle 1\rangle}, \ldots, x^{\langle T\rangle}) = \prod_t P(x^{\langle t\rangle} \mid x^{\langle 1\rangle}, \ldots, x^{\langle t-1\rangle})$;
   the RNN's hidden state supplies each factor, turning "score a whole sequence"
   into "predict the next character, repeatedly".
2. Training the model to predict the next character *is* maximising the
   probability it assigns to real text: the summed cross-entropy equals the
   negative log-likelihood of the sequence, so a good predictor is by
   construction a good generator, read forward.
3. Per-character cross-entropy in nats is reported so the figure is comparable
   across lengths; perplexity, its exponential, reads as the model's effective
   branching factor, and the $\approx \ln V$ (perplexity $\approx V$) value at
   initialisation is the "knows nothing" baseline that anchors correctness.
4. Sampling runs the predictor as a generator: draw from the distribution rather
   than take its argmax, feed the draw back as the next one-hot input, and stop on
   `\n` or a hard cap $L$ — with a seeded RNG so the string is reproducible.
5. Temperature $T$ divides the logits before the softmax, sharpening toward argmax
   as $T \to 0^{+}$ and flattening toward uniform as $T \to \infty$, trading
   faithfulness against diversity while never changing which character is most
   likely.
6. A char-LM inherits the vanilla RNN's exploding gradient over long unrolled
   sequences, so global-norm clipping — which caps the step length while
   preserving its direction — is what makes training converge at all; it does
   nothing for vanishing gradients, which need gated cells.
7. The topic introduces no new from-scratch math — the cell and its BPTT belong
   to the RNN topic — so it is realised once, canonically, in PyTorch
   (`torch.nn.RNN`, `F.cross_entropy`, `clip_grad_norm_`, `torch.multinomial`),
   checked on its own $\ln V$ anchor rather than a NumPy parity reference.
