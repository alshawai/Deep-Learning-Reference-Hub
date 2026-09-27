# Language modeling and sampling

This page is the lookup reference for the character-level recurrent language
model: the autoregressive factorisation it computes, the shifted-target training
setup and its per-character cross-entropy loss, perplexity and the
initialisation anchor, the temperature-scaled sampling algorithm, the
global-norm gradient clipping applied during training, the canonical PyTorch
primitive mapping, and the fixture and tolerances the implementation is held to.
It states the equations and makes no argument for them.

Notation extends the
[recurrent neural networks reference](recurrent-neural-networks.md) verbatim —
same $a^{\langle t\rangle}$, $x^{\langle t\rangle}$, $\hat{y}^{\langle t\rangle}$,
the same feature-first shape convention, and the same softmax read-out and
cross-entropy loss — and follows Andrew Ng's Deep Learning Specialization,
Course 5, Week 1 (Ng, 2018). The cell is the vanilla RNN of that reference; this
page adds only what the language model puts on top of it. The reasoning behind
these mechanics — why the factorisation trains a generator, why clipping is
needed here, and why temperature works — is in this topic's explanation view,
and the in-project task is in its how-to guide, both reachable from the
[Crosswalk](#crosswalk).

## Contents

- [Notation and shapes](#notation-and-shapes)
- [The language model](#the-language-model)
- [Training target and loss](#training-target-and-loss)
- [Perplexity and the initialisation anchor](#perplexity-and-the-initialisation-anchor)
- [Sampling](#sampling)
- [Gradient clipping](#gradient-clipping)
- [Framework primitives](#framework-primitives)
- [Fixture and tolerances](#fixture-and-tolerances)
- [Failure modes](#failure-modes)
- [Implementation Examples](#implementation-examples)
- [References](#references)
- [Crosswalk](#crosswalk)
- [Key Takeaways](#key-takeaways)

## Notation and shapes

The symbols the language model adds on top of the
[RNN reference](recurrent-neural-networks.md#notation-and-shapes):

| Symbol | Meaning |
| --- | --- |
| $V$ | vocabulary size: the count of unique characters in the corpus, including the terminating newline `\n` |
| `char_to_ix`, `ix_to_char` | index maps built from the **sorted** unique characters, so the encoding is identical on every machine |
| $n_x = n_y = V$ | input and output dimensions: every input and target is a one-hot column of dimension $V$ |
| $z^{\langle t\rangle}$ | read-out logits at step $t$: $z^{\langle t\rangle} = W_{ya}\,a^{\langle t\rangle} + b_y$, so $\hat{y}^{\langle t\rangle} = \mathrm{softmax}(z^{\langle t\rangle})$ |
| $T$ | the sequence length in the factorisation; the **temperature** ($T > 0$) in the sampling section — the same symbol, scoped by section |
| $L$ | the hard cap on the number of characters a single sample may emit |

All other symbols — $a^{\langle t\rangle}$, $W_{ya}$, $b_y$, and the feature-first
shapes — carry the meaning fixed in the RNN reference. Because the model is
character-level, the read-out width equals the vocabulary, $n_y = V$.

## The language model

Character-level. The corpus defines the vocabulary: $V$ is the **sorted** set of
unique characters, including the newline `\n` that terminates each sequence. The
maps `char_to_ix` / `ix_to_char` follow that sorted order, so the encoding is
deterministic across machines — a correctness requirement, not a convenience.
Every input and target is a one-hot column of dimension $V$.

The model factorises the joint probability of a length-$T$ sequence
autoregressively (Bengio, Ducharme, Vincent & Jauvin, 2003; Mikolov et al.,
2010):

$$P\big(x^{\langle 1\rangle}, \ldots, x^{\langle T\rangle}\big) = \prod_{t=1}^{T} P\big(x^{\langle t\rangle} \mid x^{\langle 1\rangle}, \ldots, x^{\langle t-1\rangle}\big)$$

The RNN read-out supplies each conditional factor:
$\hat{y}^{\langle t\rangle} = \mathrm{softmax}\big(W_{ya}\,a^{\langle t\rangle} + b_y\big)$
is the model's distribution over the next character $x^{\langle t+1\rangle}$,
conditioned on every earlier character through the hidden state
$a^{\langle t\rangle}$. This is the character-level RNN language model of
Sutskever, Martens & Hinton (2011), in the practitioner form popularised by
Karpathy (2015). Word-level modelling swaps the character vocabulary for a word
vocabulary and embeddings but keeps the same factorisation; it is out of scope
here.

## Training target and loss

**Shifted targets.** The target at each step is the next input,
$y^{\langle t\rangle} = x^{\langle t+1\rangle}$, and the final target is the
terminating newline `\n`. The first input $x^{\langle 1\rangle}$ is the **zero
vector** (there is no prior character), and the initial hidden state is
$a^{\langle 0\rangle} = \mathbf{0}$, as in the RNN reference.

**Loss.** Softmax cross-entropy in **nats** (natural logarithm). The family's
total loss is summed over timesteps and batch-averaged (see the
[RNN reference](recurrent-neural-networks.md#loss)); the language model reports
the **per-character cross-entropy**, the mean over all predicted tokens
(batch $\times$ time):

$$\mathrm{CE} = \frac{1}{T}\sum_{t=1}^{T} L^{\langle t\rangle} = -\frac{1}{m\,T}\sum_{t=1}^{T}\sum_{i=1}^{m}\sum_{c=1}^{V} y^{\langle t\rangle}_{c,i}\,\ln \hat{y}^{\langle t\rangle}_{c,i}$$

This equals the family's summed-over-$t$, batch-mean loss divided by $T$.
Reporting per character, rather than per sequence, makes the figure comparable
across sequences of different length.

## Perplexity and the initialisation anchor

**Perplexity** is the exponential of the per-character cross-entropy:

$$\mathrm{PPL} = \exp(\mathrm{CE})$$

It reads as the model's effective branching factor — the number of equally
likely next characters the model is, on average, choosing among (Goodfellow,
Bengio & Courville, 2016, Ch. 10). Lower is better; the floor is $1$ (a model
certain of every next character).

**Initialisation anchor.** Before training the read-out is approximately uniform
over the $V$ characters, so each conditional probability is $\approx 1/V$ and

$$\mathrm{CE} \approx -\ln\tfrac{1}{V} = \ln V, \qquad \mathrm{PPL} \approx V.$$

A freshly initialised model whose per-character cross-entropy is finite and
$\approx \ln V$ is behaving correctly; a figure far from $\ln V$ signals a bug in
the read-out, the loss reduction, or the vocabulary. This anchor stands in for
the numeric parity check a from-scratch NumPy reference would provide (see
[Fixture and tolerances](#fixture-and-tolerances)).

## Sampling

Sampling generates a novel sequence one character at a time, autoregressively,
feeding each drawn character back as the next input. It is **stochastic** and
**seeded**, so a sample is exactly reproducible. From the read-out logits
$z^{\langle t\rangle}$ at step $t$:

$$p^{\langle t\rangle} = \mathrm{softmax}\!\big(z^{\langle t\rangle} / T\big), \qquad i^{\langle t\rangle} \sim \mathrm{Multinomial}\big(p^{\langle t\rangle}\big)$$

The index $i^{\langle t\rangle}$ is drawn from an **explicitly-passed, seeded**
random generator and becomes the **one-hot** input $x^{\langle t+1\rangle}$ for
the next step (Graves, 2013).

**Temperature** $T > 0$ rescales the logits before the softmax:

| $T$ | Effect on $p^{\langle t\rangle}$ |
| --- | --- |
| $T = 1$ | the model's own distribution, $\mathrm{softmax}(z^{\langle t\rangle})$ |
| $T \to 0^{+}$ | collapses to the argmax (deterministic, greedy) |
| $T > 1$ | flattens toward uniform (more diverse, less faithful) |

**Stopping.** Generation stops when the newline `\n` is drawn **or** a hard cap
of $L$ characters is reached. The cap guarantees termination even from an
untrained model, whose distribution may never place enough mass on `\n`.

The framework-agnostic code for this loop lives in the explanation view (the one
place the mechanism appears as runnable code; see the [Crosswalk](#crosswalk));
this page states only the algorithm.

## Gradient clipping

Training a recurrent language model uses **global-norm gradient clipping** to
control the exploding-gradient failure mode of the vanilla RNN (Pascanu, Mikolov
& Bengio, 2013). One L2 norm is taken over every parameter gradient stacked into
a single vector, and, if it exceeds the threshold $v$ (`max_norm`), every
gradient is rescaled by the same factor — capping the magnitude while preserving
direction:

$$\lVert g\rVert_2 = \sqrt{\textstyle\sum_{\theta}\lVert d\theta\rVert_2^2}, \qquad g \leftarrow \begin{cases} \dfrac{v}{\lVert g\rVert_2}\,g & \text{if } \lVert g\rVert_2 > v, \\[4pt] g & \text{otherwise.} \end{cases}$$

This is the same operation as the `recurrent-neural-networks` topic's
from-scratch `clip_gradients`; its derivation and the vanishing/exploding
analysis are in that topic's
[reference](recurrent-neural-networks.md#vanishing-and-exploding-gradients).
Clipping addresses only the exploding case.

## Framework primitives

PyTorch is this topic's canonical — and only — implementation, so the equations
above are the contract the module honours; there is no from-scratch NumPy
module. Each mechanism maps to one framework primitive:

| Mechanism (this page) | PyTorch primitive |
| --- | --- |
| cell recurrence $a^{\langle t\rangle} = \tanh(\cdots)$ | `torch.nn.RNN` |
| read-out $z^{\langle t\rangle} = W_{ya}\,a^{\langle t\rangle} + b_y$ | `nn.Linear` |
| per-character cross-entropy (natural-log, mean over tokens) | `torch.nn.functional.cross_entropy` |
| global-norm clipping | `torch.nn.utils.clip_grad_norm_` |
| temperature-scaled multinomial draw | `torch.multinomial` with a seeded `torch.Generator` |

`F.cross_entropy` uses the natural logarithm with a mean reduction over tokens —
exactly the per-character cross-entropy defined above — and `clip_grad_norm_`
implements the global-norm rule above. The `torch.nn.RNN` parameter mapping and
its two-bias convention are tabulated in the
[RNN reference](recurrent-neural-networks.md#pytorch-parity). There is **no
generated API page** for this module: the docs build installs no framework, so
mkdocstrings cannot import it, and the link in
[Implementation Examples](#implementation-examples) points at the source on
GitHub instead.

## Fixture and tolerances

The implementation and its tests share one deterministic fixture, built by
`make_fixture()`. A language model's dimensions are set by its corpus, so the
shapes differ from the family's cell fixtures.

- **Corpus:** a fixed literal committed inside `make_fixture` — roughly ten
  short lowercase words, one per line, newline-terminated. It is never an
  external file, so it cannot drift.
- **Vocabulary:** the sorted unique characters of the corpus (including `\n`),
  giving $V = |\text{vocab}|$, $n_x = n_y = V$, and `char_to_ix` by sorted order.
- **Dimensions:** hidden size $n_a = 32$. Sequences are processed **one word at
  a time** — no padding or masking; batched-and-masked training is a how-to
  note, not the fixture.
- **Seed:** `torch.manual_seed(1)` for the weights; a fixed `torch.Generator`
  seeded `1` for sampling, passed explicitly.
- **dtype:** `float32` for training and sampling; the clip cross-check loads the
  shared gradient dict as `float64` for a tight comparison.

| Check | Asserts | Tolerance |
| --- | --- | --- |
| Init anchor | per-character cross-entropy at initialisation is finite and $\approx \ln V$ (perplexity $\approx V$) | declared `atol` |
| Training smoke | the loss **strictly decreases** over a fixed number of steps | direction, not a curve match |
| Sampling reproducibility | a sampled string is **byte-for-byte identical** under the fixed generator and a fixed $T$ | exact |
| Discrimination | the loss/perplexity **rejects a shuffled-target control** | control must score worse |
| Clip cross-check | `clip_grad_norm_` matches the NumPy `clip_gradients` on a shared gradient dict | relative error $< 10^{-10}$ (float64) |

There is no NumPy reference for this topic, so the module is checked on its
**own correctness** — the anchors above — rather than against a from-scratch
implementation.

## Failure modes

| Symptom | Cause | Fix |
| --- | --- | --- |
| Per-character cross-entropy at init far from $\ln V$ | wrong vocabulary size, a loss not mean-reduced over tokens, or a logarithm base other than $e$ | confirm $n_y = V$, natural-log cross-entropy, mean over batch $\times$ time |
| Sampling never terminates | an untrained model rarely draws `\n`, and there is no hard cap | enforce the cap $L$ as a stop condition alongside `\n` |
| Sampled string differs between runs | the RNG is not seeded, or the generator is not passed explicitly | draw from an explicitly-passed, seeded `torch.Generator` |
| Encoding differs across machines | the vocabulary was built from an unordered set | build `char_to_ix` from the **sorted** unique characters |
| Training loss diverges to NaN or Inf | exploding gradients, left unclipped | apply global-norm clipping with a finite `max_norm` |
| $T = 0$ raises or yields NaNs | division by zero in $z^{\langle t\rangle}/T$ | require $T > 0$; approach the greedy limit with a small $T$, or special-case the argmax |

## Implementation Examples

### Complete Implementations:
> #### **[Character-level language model — PyTorch (canonical)](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/pytorch/sequence/language_model.py)** - The char-RNN language model on `torch.nn.RNN` + `nn.Linear`, with per-character cross-entropy and perplexity through `F.cross_entropy`, global-norm clipping through `clip_grad_norm_`, and seeded temperature sampling through `torch.multinomial`. Checked on its own correctness against the shared fixture: the $\approx \ln V$ initialisation anchor, a strictly decreasing training loss, byte-for-byte sampling reproducibility, a shuffled-target discrimination test, and a `clip_grad_norm_`-versus-`clip_gradients` cross-check. This is the hub's first framework-canonical topic; the docs build installs no framework, so there is no generated API page.

## References

- **Bengio, Y., Ducharme, R., Vincent, P. & Jauvin, C. (2003). A Neural Probabilistic Language Model. _Journal of Machine Learning Research_ 3, 1137–1155.** – The neural language-model factorisation this page states.
- **Mikolov, T., Karafiát, M., Burget, L., Černocký, J. & Khudanpur, S. (2010). Recurrent neural network based language model. INTERSPEECH.** – The RNN realisation of that factorisation.
- **Sutskever, I., Martens, J. & Hinton, G. (2011). Generating Text with Recurrent Neural Networks. ICML.** – Character-level generation from a recurrent network.
- **Graves, A. (2013). Generating Sequences With Recurrent Neural Networks.**  
  [https://arxiv.org/abs/1308.0850](https://arxiv.org/abs/1308.0850) – Sampling and sequence generation, including the temperature-scaled draw.
- **Karpathy, A. (2015). The Unreasonable Effectiveness of Recurrent Neural Networks.**  
  [https://karpathy.github.io/2015/05/21/rnn-effectiveness/](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) – The char-RNN language model and sampling, in practitioner form.
- **Pascanu, R., Mikolov, T. & Bengio, Y. (2013). On the difficulty of training Recurrent Neural Networks. ICML.**  
  [https://arxiv.org/abs/1211.5063](https://arxiv.org/abs/1211.5063) – Exploding gradients and global-norm clipping.
- **Goodfellow, I., Bengio, Y. & Courville, A. (2016). _Deep Learning_, Ch. 10.**  
  [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/) – Cross-entropy and perplexity as the language-model training signal.
- **Ng, A. (2018). Deep Learning Specialization, Course 5 (Sequence Models), Week 1.**  
  [https://www.coursera.org/specializations/deep-learning](https://www.coursera.org/specializations/deep-learning) – Source of the notation used here, and the _Dinosaur Island_ character-level assignment this topic realises.

## Crosswalk

Live links are added as each sibling ships; plain-text entries are not yet
published.

| Reader wants to | Go to |
| --- | --- |
| Learn it step by step | [Language modeling and sampling (notebook)](../tutorials/03-language-modeling-and-sampling.ipynb) |
| Do it in a project | [Train a language model](../how-to/train-a-language-model.md) |
| Look up the factorisation or sampling algorithm | Language modeling and sampling — *this page* |
| Understand why it works | [Language modeling and sampling](../explanation/language-modeling-and-sampling.md) |
| Learn by running it | [Language modeling and sampling (notebook)](../tutorials/03-language-modeling-and-sampling.ipynb) |

## Key Takeaways

1. The language model factorises a sequence autoregressively, $P(x^{\langle 1\rangle}, \ldots, x^{\langle T\rangle}) = \prod_{t} P(x^{\langle t\rangle} \mid x^{\langle 1\rangle}, \ldots, x^{\langle t-1\rangle})$, and the RNN's softmax read-out $\hat{y}^{\langle t\rangle} = \mathrm{softmax}(W_{ya} a^{\langle t\rangle} + b_y)$ supplies each conditional as the distribution over the next character.
2. Character-level: the vocabulary $V$ is the sorted unique characters (including `\n`), so $n_x = n_y = V$ and every token is a one-hot column; the sorted order makes the encoding deterministic across machines.
3. Targets are shifted by one, $y^{\langle t\rangle} = x^{\langle t+1\rangle}$, with $x^{\langle 1\rangle} = \mathbf{0}$; the loss is the per-character cross-entropy in nats, and perplexity is its exponential, $\mathrm{PPL} = \exp(\mathrm{CE})$.
4. At initialisation the model is $\approx$ uniform, so the per-character cross-entropy is $\approx \ln V$ and perplexity is $\approx V$ — the implementation's correctness anchor in place of a NumPy parity check.
5. Sampling is autoregressive, stochastic, and seeded: $p^{\langle t\rangle} = \mathrm{softmax}(z^{\langle t\rangle}/T)$, draw $i^{\langle t\rangle}$, feed it back as a one-hot input; $T = 1$ is the model's own distribution, $T \to 0^{+}$ is greedy, $T > 1$ diversifies, and generation stops on `\n` or the cap $L$.
6. Global-norm clipping caps the exploding-gradient case during training, rescaling every gradient by $v / \lVert g\rVert_2$ when the norm exceeds $v$, while leaving direction unchanged.
7. PyTorch is canonical: `torch.nn.RNN` + `nn.Linear`, `F.cross_entropy`, `clip_grad_norm_`, and a seeded `torch.multinomial`, checked on its own correctness against a deterministic corpus fixture ($n_a = 32$, seed `1`).
