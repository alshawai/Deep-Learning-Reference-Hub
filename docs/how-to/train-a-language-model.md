# How to train a language model and generate text

You have a corpus of text and you want a model that writes more of it — novel
sequences in the same style, produced one character at a time. This guide takes
a raw corpus to a trained character-level RNN language model and a
temperature-controlled sampler in PyTorch: building the vocabulary, shifting the
targets, training with global-norm gradient clipping, reading perplexity as your
signal, and sampling reproducibly.

It assumes you already know the vanilla RNN cell and backpropagation through
time — the recurrence $a^{\langle t\rangle} = \tanh(W_{aa}\,a^{\langle t-1\rangle} + W_{ax}\,x^{\langle t\rangle} + b_a)$
and the softmax read-out are taken as given here. If they are not yet given,
start from the [recurrent neural networks reference](../reference/recurrent-neural-networks.md)
and its [explanation](../explanation/recurrent-neural-networks.md). The
factorisation and the sampling algorithm are stated in lookup form in this
topic's reference view, and the reasons they work are argued in its explanation
view; both are reachable from the [Crosswalk](#crosswalk).

## Contents

- [Turn your corpus into a vocabulary](#turn-your-corpus-into-a-vocabulary)
- [Build the model](#build-the-model)
- [Shift the targets and choose the loss](#shift-the-targets-and-choose-the-loss)
- [Sanity-check the loss before you train](#sanity-check-the-loss-before-you-train)
- [Train with global-norm gradient clipping](#train-with-global-norm-gradient-clipping)
- [Sample novel sequences with a temperature knob](#sample-novel-sequences-with-a-temperature-knob)
- [Scale up to batches and word-level models](#scale-up-to-batches-and-word-level-models)
- [Diagnose a model that will not learn](#diagnose-a-model-that-will-not-learn)
- [Implementation Examples](#implementation-examples)
- [References](#references)
- [Crosswalk](#crosswalk)
- [Key Takeaways](#key-takeaways)

## Turn your corpus into a vocabulary

Character-level modeling makes the vocabulary the easy part: it is exactly the
set of distinct characters your corpus contains. Sort that set so the mapping is
**deterministic across machines** — the same corpus always yields the same
integer for the same character — and include the newline `\n`, because it is the
token that ends a generated sequence.

```python
corpus = ["cat\n", "dog\n", "bird\n"]        # newline-terminated sequences
text = "".join(corpus)
vocab = sorted(set(text))                     # sorted -> deterministic mapping
V = len(vocab)                                # n_x = n_y = V
char_to_ix = {ch: i for i, ch in enumerate(vocab)}
ix_to_char = {i: ch for i, ch in enumerate(vocab)}
```

Every input and target is a one-hot vector of dimension `V`, so the input
dimension and the output dimension are both the vocabulary size:
$n_x = n_y = V$. That single fact drives every shape below.

Character-level is the right default when the alphabet is small and closed (a
name generator, code, a single language's prose): the model can spell any string
and never meets an out-of-vocabulary token. Move to **word-level or subword**
tokenisation when the vocabulary is large or open — see
[Scale up to batches and word-level models](#scale-up-to-batches-and-word-level-models).

## Build the model

The language model factorises the joint probability of a sequence
autoregressively,

$$P(x^{\langle 1\rangle}, \dots, x^{\langle T\rangle}) = \prod_{t=1}^{T} P\big(x^{\langle t\rangle} \mid x^{\langle 1\rangle}, \dots, x^{\langle t-1\rangle}\big),$$

and the RNN supplies each conditional: the read-out
$\hat{y}^{\langle t\rangle} = \mathrm{softmax}(W_{ya}\,a^{\langle t\rangle} + b_y)$
is the distribution over the next character (Bengio et al., 2003; Mikolov et
al., 2010). In PyTorch that is a `torch.nn.RNN` for the tanh recurrence and an
`nn.Linear` for the read-out.

```python
import torch


class CharRNN(torch.nn.Module):
    """A character-level RNN language model: a tanh recurrence and a read-out."""

    def __init__(self, V: int, n_a: int) -> None:
        super().__init__()
        self.rnn = torch.nn.RNN(input_size=V, hidden_size=n_a, nonlinearity="tanh")
        self.readout = torch.nn.Linear(n_a, V)

    def forward(self, x, a0=None):
        a, a_last = self.rnn(x, a0)   # a: (T, m, n_a) -- PyTorch's (seq, batch, feature)
        logits = self.readout(a)      # (T, m, V) -- one score per character, per step
        return logits, a_last
```

Two conventions of `torch.nn.RNN` differ from the hub's from-scratch notation
and bite the first time you wire it up. It speaks `(seq, batch, feature)`, not
the reference's `(feature, batch, seq)`; and it carries **two** bias vectors
(`b_ih` and `b_hh`) where the reference cell has one. Neither changes the math
here — the two biases only ever appear as their sum — but they matter the moment
you transplant weights between the two. Passing `a0=None` uses the zero initial
state $a^{\langle 0\rangle} = 0$, which is what you want.

The hidden size `n_a` is your main capacity knob. Keep it small (`32`) for a
tiny corpus that must train in seconds; reach for `128`–`512` on real text. The
read-out returns raw **logits**, not probabilities — the loss below wants logits,
and applying `softmax` here would double-count it.

## Shift the targets and choose the loss

The training signal is the corpus itself, shifted by one position: at every step
the target is the **next** character, $y^{\langle t\rangle} = x^{\langle t+1\rangle}$.
The first input $x^{\langle 1\rangle}$ is the zero vector — there is no prior
character — and the final target is the terminating `\n`, which is how the model
learns where a sequence ends.

```python
import torch


def make_example(word: str, char_to_ix: dict[str, int], V: int):
    """Shifted (input, target) pair for one newline-terminated word.

    inputs : (T, 1, V) one-hot, with x^{<1>} the zero vector.
    targets: (T,) class indices, y^{<t>} = x^{<t+1>}, ending in the newline.
    """
    idx = [char_to_ix[ch] for ch in word]     # e.g. c, a, t, \n
    targets = torch.tensor(idx)
    inputs = torch.zeros(len(idx), 1, V)       # x^{<1>} is the zero vector
    for t, prev in enumerate(idx[:-1], start=1):
        inputs[t, 0, prev] = 1.0               # x^{<t+1>} is character t, one-hot
    return inputs, targets
```

For the word `"cat\n"` this pairs the zero vector with target `c`, then `c` with
`a`, `a` with `t`, and `t` with `\n` — the model predicts each character from the
ones before it, and predicts the newline once the word is spelled.

The loss is softmax cross-entropy, and the idiomatic tool is `F.cross_entropy`
applied to the **raw logits**. It fuses the softmax and the negative log
likelihood in a numerically stable way (never `softmax` then `log` at training
time), computes in **nats** (natural log), and with the default `mean` reduction
returns the **per-character cross-entropy** — the mean over every predicted
token. **Perplexity** is its exponential:

```python
import torch.nn.functional as F

logits, _ = model(inputs)                     # (T, 1, V)
loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
perplexity = loss.exp()                        # exp(per-character cross-entropy)
```

Per-character cross-entropy is the standard language-model metric because it is
length-independent: it measures how surprised the model is by the average
character, and perplexity re-expresses that as an effective branching factor —
"the model is as uncertain as if it were choosing uniformly among this many
characters" (Bengio et al., 2003).

## Sanity-check the loss before you train

Before you spend any compute, check the loss at initialisation. An untrained
model with small random weights is approximately **uniform** over the `V`
characters, so its per-character cross-entropy should be close to $\ln V$ and
its perplexity close to `V`:

```python
import math

import torch
import torch.nn.functional as F

model = CharRNN(V, n_a=32)
inputs, targets = make_example("cat\n", char_to_ix, V)

with torch.no_grad():
    logits, _ = model(inputs)
    loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))

print(loss.item(), math.log(V))   # per-character cross-entropy vs. ln V -- close at init
```

If the step-0 loss is far from $\ln V$, stop and find the wiring bug before you
train — it is almost always one of: a `softmax` applied before `cross_entropy`
(counting the softmax twice), the wrong reduction, or targets that are off by one
position. One more check worth keeping: the loss should be able to **reject a
control**. Shuffle the targets and confirm the loss rises; a metric that a
scrambled target cannot move is not measuring what you think it is.

## Train with global-norm gradient clipping

The training loop is the ordinary one, with a single non-negotiable addition:
**clip the gradients by their global norm** on every step. A vanilla RNN
backpropagates through time by repeatedly multiplying by the same recurrent
Jacobian, so gradients can grow geometrically and produce a `NaN` loss in one
bad step. Global-norm clipping takes the L2 norm over **all** gradients viewed as
one vector and rescales them if that norm exceeds `max_norm`, preserving their
direction (Pascanu, Mikolov & Bengio, 2013). `torch.nn.utils.clip_grad_norm_` is
that operation.

```python
import math

import torch
import torch.nn.functional as F

torch.manual_seed(1)
model = CharRNN(V, n_a=32)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

num_steps = 2000
for step in range(num_steps):
    word = corpus[step % len(corpus)]
    inputs, targets = make_example(word, char_to_ix, V)

    logits, _ = model(inputs)
    loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)   # after backward, before step
    optimizer.step()

    if step % 200 == 0:
        print(step, loss.item(), math.exp(loss.item()))   # step, loss, perplexity
```

The one ordering that matters: clip **after** `loss.backward()` has populated the
gradients and **before** `optimizer.step()` consumes them. Clip before the
backward pass and you clip last step's gradients; clip after the step and you
clip nothing that mattered. A `max_norm` around `5.0` is a good starting point.
On this three-word corpus the per-character loss should fall steadily and
perplexity should drop toward `1`; on a real corpus watch for a loss that
decreases at all before you tune anything else. Clipping addresses only the
**exploding** half of the RNN gradient problem — if long-range structure never
gets learned, that is vanishing gradients, and the fix is a gated cell, not a
larger `max_norm` (see [Diagnose a model that will not learn](#diagnose-a-model-that-will-not-learn)).

## Sample novel sequences with a temperature knob

A trained model generates by running the factorisation forward: draw a
character from the read-out distribution, feed it back in as the next input, and
repeat until you draw the newline (Graves, 2013). Two decisions make this
practitioner-grade rather than a toy loop — a **temperature** to control how
adventurous the draws are, and an **explicit seeded generator** so a sample is
exactly reproducible.

```python
import torch
import torch.nn.functional as F


def sample(model, char_to_ix, ix_to_char, V,
           temperature=1.0, max_length=50, seed=1):
    """Autoregressively sample one newline-terminated string.

    Reproducible: draws come from an explicitly-seeded generator, so a fixed
    (seed, temperature) always returns the same string. Requires temperature > 0.
    """
    generator = torch.Generator().manual_seed(seed)
    newline = char_to_ix["\n"]

    x = torch.zeros(1, 1, V)             # x^{<1>} is the zero vector
    a = None
    chars = []
    for _ in range(max_length):
        logits, a = model(x, a)          # thread the hidden state across steps
        probs = F.softmax(logits[-1, 0] / temperature, dim=-1)
        idx = torch.multinomial(probs, num_samples=1, generator=generator).item()
        if idx == newline:
            break
        chars.append(ix_to_char[idx])
        x = torch.zeros(1, 1, V)         # feed the drawn character back in
        x[0, 0, idx] = 1.0
    return "".join(chars)
```

**Temperature** $T$ scales the logits before the softmax, $p = \mathrm{softmax}(z / T)$:

- $T = 1$ samples from the model's own distribution.
- $T \to 0^{+}$ sharpens toward the arg-max — repetitive, "safe" text. (Pass a
  small value such as `0.5`, not `0`, which would divide by zero.)
- $T > 1$ flattens toward uniform — more surprising, less coherent text.

```python
for temperature in (0.5, 1.0, 1.5):
    print(temperature, sample(model, char_to_ix, ix_to_char, V, temperature=temperature))
```

Two details keep the loop honest. **Thread the hidden state**: each call passes
the previous step's `a` back in, so the model conditions on everything sampled so
far rather than restarting. And **cap the length** with `max_length`: an
untrained model may never draw the newline, and the cap guarantees the loop
terminates anyway. Because the draws come from the passed-in `generator`, a fixed
`(seed, temperature)` reproduces a string byte-for-byte; reseed when you want
variety.

## Scale up to batches and word-level models

The loop above trains one sequence at a time, which is the clearest place to
start and exactly how the hub's fixture is built. Two extensions cover most real
corpora.

**Batch with padding and a mask.** Pad the sequences in a batch to a common
length, and make the padded positions contribute nothing to the loss. The clean
way is to pad the targets with a sentinel and hand it to `cross_entropy` as
`ignore_index`:

```python
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# input_sequences: list of (T_i, V) one-hot tensors; target_sequences: (T_i,) index tensors.
padded_inputs = pad_sequence(input_sequences)                      # (T_max, batch, V)
padded_targets = pad_sequence(target_sequences, padding_value=-1)  # (T_max, batch)

logits, _ = model(padded_inputs)                                   # (T_max, batch, V)
loss = F.cross_entropy(
    logits.reshape(-1, V),
    padded_targets.reshape(-1),
    ignore_index=-1,          # padded steps are skipped, not scored as a class
)
```

For long, ragged batches, `torch.nn.utils.rnn.pack_padded_sequence` lets the RNN
skip padded steps entirely rather than computing and then masking them.

**Go word-level.** When the alphabet is large or open, swap the character
vocabulary for a word or subword tokeniser and replace the one-hot input with an
`nn.Embedding` lookup. Everything else here is unchanged: the shifted targets,
the per-token cross-entropy and perplexity, the clipping, and the temperature
sampler all carry over verbatim. Word-level modeling and embeddings are their own
subject and are not implemented in this topic.

As of 2026 the production default for language modeling is the subword
Transformer, not the recurrent cell shown here — but the objective (predict the
next token) and the sampler (temperature over a softmax) are the very same
mechanism, which is why this loop is worth understanding before you scale it up.

## Diagnose a model that will not learn

| Symptom | First thing to check |
| --- | --- |
| Loss goes `NaN` or diverges | Clip by global norm every step, after `backward()` and before `step()`. A vanilla RNN explodes without it (Pascanu, Mikolov & Bengio, 2013). |
| Step-0 loss is far from $\ln V$ | A wiring bug: `softmax` applied before `cross_entropy`, the wrong reduction, or targets off by one position. |
| Loss falls, then flattens; long-range structure never learned | Vanishing gradients, which clipping does not fix. Reach for a gated cell — see [choose a gated cell](choose-a-gated-cell.md). |
| Samples are gibberish even at low temperature | Undertrained, `n_a` too small, or the learning rate too low. Confirm the loss is actually falling first. |
| Samples never terminate | `\n` must be in the vocabulary and be a real target; keep the `max_length` cap regardless. |
| The same seed gives a different sample each run | You are drawing from the global RNG. Pass an explicit `torch.Generator` and draw from it. |
| Perplexity looks great but text is bland | Temperature too low — you are near the arg-max. Raise it toward `1.0`. |

## Implementation Examples

### Complete Implementations:

> #### **[Character-level language model — PyTorch](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/pytorch/sequence/language_model.py)** - The char-RNN language model built on `torch.nn.RNN` + `nn.Linear`, with `F.cross_entropy` in nats for the loss, `torch.nn.utils.clip_grad_norm_` for global-norm clipping, and a seeded `torch.multinomial` sampler. This topic's canonical implementation — the hub's first framework-canonical module, checked on its own correctness: the $\ln V$ initialisation anchor, a strictly decreasing training loss, byte-for-byte reproducible sampling, a shuffled-target discrimination control, and a `clip_grad_norm_`-versus-NumPy cross-check on a shared gradient dictionary.

## References

- **Bengio, Y., Ducharme, R., Vincent, P. & Jauvin, C. (2003). A Neural Probabilistic Language Model. _Journal of Machine Learning Research_ 3, 1137–1155.** – The neural language-model factorisation and the perplexity objective this guide trains.
- **Mikolov, T., Karafiát, M., Burget, L., Černocký, J. & Khudanpur, S. (2010). Recurrent neural network based language model. INTERSPEECH 2010.** – The RNN language model, of which this character-level model is an instance.
- **Sutskever, I., Martens, J. & Hinton, G. (2011). Generating Text with Recurrent Neural Networks. ICML 2011.** – Character-level text generation from a recurrent net.
- **Graves, A. (2013). Generating Sequences With Recurrent Neural Networks.**  
  [https://arxiv.org/abs/1308.0850](https://arxiv.org/abs/1308.0850) – Autoregressive sampling and sequence generation, including the temperature knob.
- **Karpathy, A. (2015). The Unreasonable Effectiveness of Recurrent Neural Networks.**  
  [https://karpathy.github.io/2015/05/21/rnn-effectiveness/](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) – The char-RNN language model and sampling, in practitioner form.
- **Pascanu, R., Mikolov, T. & Bengio, Y. (2013). On the difficulty of training Recurrent Neural Networks. ICML.**  
  [https://arxiv.org/abs/1211.5063](https://arxiv.org/abs/1211.5063) – Exploding gradients and global-norm clipping, the remedy applied here.
- **Goodfellow, I., Bengio, Y. & Courville, A. (2016). _Deep Learning_, Ch. 10.**  
  [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/) – Sequence modeling, language models, and the vanishing/exploding-gradient treatment.
- **Ng, A. (2018). Deep Learning Specialization, Course 5 (Sequence Models), Week 1.**  
  [https://www.coursera.org/specializations/deep-learning](https://www.coursera.org/specializations/deep-learning) – The notation used here, and the Dinosaur-Island character-LM assignment this guide generalises.

## Crosswalk

Live links are added as each sibling ships; entries not yet published render as
plain text.

| Reader wants to | Go to |
| --- | --- |
| Learn it step by step | Language modeling and sampling (notebook) — *ships with the framework-notebook toolchain* |
| Do it in a project | Train a language model — *this page* |
| Look up the factorisation or sampling algorithm | [Language modeling and sampling](../reference/language-modeling-and-sampling.md) |
| Understand why it works | [Language modeling and sampling](../explanation/language-modeling-and-sampling.md) |
| Learn by running it | Language modeling and sampling (notebook) — *ships with the framework-notebook toolchain* |

## Key Takeaways

1. A language model factorises $P(x^{\langle 1\rangle}, \dots, x^{\langle T\rangle}) = \prod_t P(x^{\langle t\rangle} \mid x^{\langle 1\rangle}, \dots, x^{\langle t-1\rangle})$,
   and a char-RNN supplies each conditional through its softmax read-out.
   Character-level means the vocabulary is the sorted unique characters, so
   $n_x = n_y = V$.
2. Train on shifted targets — $y^{\langle t\rangle} = x^{\langle t+1\rangle}$,
   the first input the zero vector, the final target the newline. Feed raw logits
   to `F.cross_entropy` (natural log, mean over tokens): that mean is the
   per-character cross-entropy, and perplexity is its exponential.
3. Sanity-check before training. At initialisation the per-character loss should
   be $\approx \ln V$ and perplexity $\approx V$. A step-0 loss far from $\ln V$
   is a wiring bug — a double softmax, the wrong reduction, or off-by-one targets.
4. Clip by global norm every step, **after** `backward()` and **before**
   `step()`, with `max_norm` around `5.0`. Clipping is the exploding-gradient
   remedy (Pascanu et al., 2013); it does nothing for vanishing gradients, which
   are a gated cell's job.
5. Sample autoregressively: $p = \mathrm{softmax}(z / T)$, draw with
   `torch.multinomial`, feed the drawn character back as the next one-hot input,
   and stop on the newline or a hard length cap. $T = 1$ is the model's own
   distribution, $T \to 0^{+}$ collapses to the arg-max, $T > 1$ flattens.
6. Reproducibility comes from an explicit seeded generator: a fixed
   `(seed, temperature)` returns the same string byte-for-byte. Reseed for
   variety.
7. Scale by padding a batch and masking the padded targets with `ignore_index`,
   and go word-level by swapping the character vocabulary for a tokeniser plus
   `nn.Embedding` — the shifted-target training and temperature sampling are
   unchanged. Transformer LMs are the modern default (as of 2026); this char-RNN
   is the mechanism they generalise.
