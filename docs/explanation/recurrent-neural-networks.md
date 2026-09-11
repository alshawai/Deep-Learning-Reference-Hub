# Recurrent neural networks

A recurrent neural network (RNN) processes a sequence one step at a time,
carrying a hidden state that summarises everything it has read so far. This page
explains *why* the architecture is built the way it is: what problem recurrence
solves that a feedforward network cannot, how one small cell is unrolled and
trained across time, why that same training makes long-range dependencies so
hard to learn, and how the bidirectional and deep variants turn out to be the
same cell wired up differently rather than new mathematics.

It assumes you already know feedforward networks, backpropagation, and the
softmax cross-entropy loss. For the equations and tensor shapes in lookup form,
and for a runnable step-by-step version, see the sibling views under
[Related documentation](#related-documentation). Throughout, the notation
follows Andrew Ng's Deep Learning Specialization, Course 5: activations
$a^{\langle t\rangle}$, inputs $x^{\langle t\rangle}$, predictions
$\hat{y}^{\langle t\rangle}$, and shared weights $W_{aa}$, $W_{ax}$, $W_{ya}$.

## Contents

- [Why sequences need recurrence](#why-sequences-need-recurrence)
- [The recurrent cell](#the-recurrent-cell)
- [Forward propagation through time](#forward-propagation-through-time)
- [Backpropagation through time](#backpropagation-through-time)
- [Why long-range dependencies are hard](#why-long-range-dependencies-are-hard)
- [Gradient clipping](#gradient-clipping)
- [One cell and many wiring patterns](#one-cell-and-many-wiring-patterns)
- [Bidirectional and deep RNNs](#bidirectional-and-deep-rnns)
- [Where RNNs stand today](#where-rnns-stand-today)
- [Key Takeaways](#key-takeaways)

## Why sequences need recurrence

A feedforward network expects a fixed-size input and produces a fixed-size
output. Sequences break both assumptions: sentences, audio, and time series vary
in length, and the meaning of an element depends on its neighbours. Forcing a
sequence through a plain feedforward network runs into three problems at once.

- **Variable length.** A fixed input layer cannot accept a five-word sentence
  and a fifty-word sentence with the same weights. Padding to a maximum length
  wastes capacity and still caps the length the model can ever handle.
- **No parameter sharing across positions.** If every position had its own
  weights, a pattern learned at position 3 — that a word is a name, say — would
  teach the network nothing about the same pattern at position 17. Sequences are
  translation-invariant in time, and the model should be too.
- **No memory.** A feedforward network sees each element in isolation. Deciding
  whether "Teddy" is a name needs the words around it.

Recurrence answers all three together. The RNN applies the *same* function at
every step (parameter sharing), threads a hidden state from one step to the next
(memory), and runs for as many steps as the sequence is long (variable length).
The hidden state is a fixed-size, learned summary of the history — a lossy
running compression of everything the model has read so far (Elman, 1990;
Goodfellow, Bengio & Courville, 2016, Ch. 10).

## The recurrent cell

At each step $t$ the cell reads the current input $x^{\langle t\rangle}$ and the
previous hidden state $a^{\langle t-1\rangle}$, and produces a new hidden state
and a read-out:

$$a^{\langle t\rangle} = \tanh\!\big(W_{aa}\,a^{\langle t-1\rangle} + W_{ax}\,x^{\langle t\rangle} + b_a\big)$$

$$\hat{y}^{\langle t\rangle} = \mathrm{softmax}\!\big(W_{ya}\,a^{\langle t\rangle} + b_y\big)$$

The initial state is the zero vector, $a^{\langle 0\rangle} = 0$. Two terms drive
the hidden update: the *recurrent* term $W_{aa}\,a^{\langle t-1\rangle}$, which
carries the past forward, and the *input* term $W_{ax}\,x^{\langle t\rangle}$,
which injects the present. The $\tanh$ squashes their sum into $(-1, 1)$, keeping
the state bounded step after step; its role in the backward pass turns out to be
central to the vanishing-gradient story below. The read-out is a separate affine
map through a softmax, turning the hidden state into a distribution over $n_y$
classes. Not every step needs a read-out — which steps do is exactly what
separates the architecture types described
[below](#one-cell-and-many-wiring-patterns).

Ng also writes the recurrence in a stacked form that folds the two weight
matrices into one:

$$a^{\langle t\rangle} = \tanh\!\big(W_{a}\,[\,a^{\langle t-1\rangle};\,x^{\langle t\rangle}\,] + b_a\big), \qquad W_{a} = [\,W_{aa}\mid W_{ax}\,]$$

The two forms are numerically identical. Keeping $W_{aa}$ and $W_{ax}$ separate
is a teaching choice: it makes the *recurrent-path* gradient and the *input-path*
gradient appear as distinct terms during backpropagation, which is the core
lesson of training an RNN. The exact tensor shape of every symbol is tabulated in
the reference view (see [Related documentation](#related-documentation)).

!!! note "Why a framework's RNN looks different"
    Production libraries factor the cell slightly differently. `torch.nn.RNN`,
    for instance, computes
    $h_t = \tanh(W_{ih}\,x_t + b_{ih} + W_{hh}\,h_{t-1} + b_{hh})$ — two bias
    vectors instead of one — and returns only the hidden states, leaving the
    $\hat{y}$ read-out to a separate `nn.Linear`. The forms are equivalent: set
    $W_{ih} = W_{ax}$, $W_{hh} = W_{aa}$, $b_{ih} = b_a$, and $b_{hh} = 0$, and
    let a linear layer carry $W_{ya}$ and $b_y$. The redundant second bias buys
    the library a more uniform, fusable set of operations.

## Forward propagation through time

Running the network over a whole sequence means applying the one cell
repeatedly, threading the hidden state forward:

$$a^{\langle 0\rangle} \to a^{\langle 1\rangle} \to a^{\langle 2\rangle} \to \cdots \to a^{\langle T_x\rangle}$$

with the same weights $W_{aa}, W_{ax}, W_{ya}, b_a, b_y$ reused at every step.
Drawing the steps side by side "unrolls" the recurrence into a feedforward graph
that is $T_x$ layers deep — but a very unusual one, because every layer shares a
single set of weights. That shared-weight, arbitrarily-deep structure is the
source of both the RNN's strength, since it generalises across positions and
lengths, and its central difficulty, which is training it.

## Backpropagation through time

Training minimises a loss summed over the steps that produce an output, averaged
over the $m$ examples in the batch:

$$L = \sum_{t} L^{\langle t\rangle}, \qquad L^{\langle t\rangle} = -\frac{1}{m}\sum_{i,\,c} y^{\langle t\rangle}_{c,i}\,\log \hat{y}^{\langle t\rangle}_{c,i}$$

Backpropagation through time (BPTT) is ordinary backpropagation applied to the
unrolled graph, walked from $t = T_x$ back to $t = 1$. Two features make it
specific to recurrence.

First, the gradient arriving at each hidden state has **two sources**. The state
$a^{\langle t\rangle}$ influences the loss through its own read-out
$\hat{y}^{\langle t\rangle}$ *and* through the next state
$a^{\langle t+1\rangle}$ it helped produce, so the total signal is the sum of the
two:

$$\frac{\partial L}{\partial a^{\langle t\rangle}} = \underbrace{\tfrac{1}{m}\,W_{ya}^{\top}\big(\hat{y}^{\langle t\rangle} - y^{\langle t\rangle}\big)}_{\text{read-out at }t} \;+\; \underbrace{W_{aa}^{\top}\Big(\big(1 - (a^{\langle t+1\rangle})^2\big)\odot \tfrac{\partial L}{\partial a^{\langle t+1\rangle}}\Big)}_{\text{recurrence from }t+1}$$

The softmax-with-cross-entropy gradient collapses neatly to
$\hat{y}^{\langle t\rangle} - y^{\langle t\rangle}$ at the logits, and the final
step $t = T_x$ has no successor, so its recurrence term is zero.

Second, because the weights are **shared** across every step, each weight
collects a gradient contribution from *every* step, and those contributions add
up:

$$\mathrm{d}W_{aa} = \sum_{t=1}^{T_x} \frac{\partial L}{\partial W_{aa}}\bigg|_{\text{step } t}, \qquad \text{and likewise for } W_{ax},\ W_{ya},\ b_a,\ b_y.$$

This accumulation is why one mislabelled step still nudges the weights that every
other step also used. The $\tanh$ backward relies on the identity
$\tanh'(z) = 1 - \tanh^2(z) = 1 - (a^{\langle t\rangle})^2$, read straight off the
cached activation rather than recomputed.

## Why long-range dependencies are hard

Follow the recurrence term backwards. Each step multiplies the incoming
hidden-state gradient by the same Jacobian,

$$\frac{\partial a^{\langle t\rangle}}{\partial a^{\langle t-1\rangle}} = \mathrm{diag}\!\big(1 - (a^{\langle t\rangle})^2\big)\, W_{aa},$$

so the gradient that reaches a state $k$ steps in the past is a *product* of $k$
such Jacobians:

$$\frac{\partial a^{\langle t\rangle}}{\partial a^{\langle t-k\rangle}} = \prod_{i=t-k+1}^{t} \mathrm{diag}\!\big(1 - (a^{\langle i\rangle})^2\big)\, W_{aa}.$$

A product of $k$ nearly-identical matrices behaves like the $k$-th power of one:
its size is governed geometrically by the largest singular value of the repeated
factor. The $\tanh$ derivative $1 - (a)^2$ is at most $1$, so the recurrent
weight $W_{aa}$ sets the scale. If its largest singular value is below $1$, the
product shrinks toward zero as $k$ grows — the gradient from distant steps
**vanishes**, and the network simply cannot feel a dependency that spans many
steps. If it is above $1$, the product can grow without bound — the gradient
**explodes**, and one update can throw the weights to infinity or `NaN`.

This is not a bug to be fixed with better code; it is a structural consequence of
multiplying the same Jacobian many times, first analysed by Bengio, Simard &
Frasconi (1994) and made precise by Pascanu, Mikolov & Bengio (2013). The two
failure modes call for different remedies. Exploding gradients have a cheap,
direct fix (below). Vanishing gradients do not — no rescaling recovers a signal
that has already decayed to zero — which is exactly why the *architecture* has to
change. Gated cells (the LSTM and the GRU) add a near-linear path along which the
gradient travels without being multiplied by $W_{aa}$ at every step; that is the
motivation for the gated-cell topic in this family, which this page deliberately
stops short of.

## Gradient clipping

Exploding gradients are tamed by *clipping*. Take the global norm of all the
parameter gradients viewed as one long vector, and if it exceeds a threshold
$\theta$, rescale every gradient by the same factor so the norm is exactly
$\theta$:

$$g \leftarrow g \cdot \min\!\Big(1,\ \frac{\theta}{\lVert g \rVert}\Big).$$

Because a single scalar multiplies all gradients at once, clipping caps the
*magnitude* of the step while leaving its *direction* unchanged — the update
still points where the gradients said to go, just not as far. This is the remedy
Pascanu, Mikolov & Bengio (2013) proposed, and it is cheap enough to leave on by
default. Two things it is not: it does nothing for vanishing gradients (a zero
gradient rescaled is still zero), and it is not a substitute for a stable
architecture. It keeps training from diverging; it does not extend the range of
dependencies the network can actually learn.

## One cell and many wiring patterns

The cell above produces one output per input, but the *same* cell supports every
input/output shape simply by choosing which steps receive an input and which emit
a read-out. This is why a course introduces a whole "zoo" of RNN architectures
and then implements only one: the rest are wiring, not new mathematics (Ng,
2018).

- **One-to-one:** a single input, a single output — the degenerate case, which
  is just a feedforward network.
- **One-to-many:** one input, a sequence out. Sequence generation works this
  way, emitting a token and feeding it back as the next input; music generation
  and image captioning fit here.
- **Many-to-one:** a sequence in, one output at the end. Sentiment
  classification reads a whole review and emits a single label, so only the last
  step has a read-out.
- **Many-to-many, equal length** ($T_x = T_y$): one output per input step,
  aligned. Named-entity recognition tags each word. This is the case the
  implementation derives in full.
- **Many-to-many, unequal length** ($T_x \neq T_y$): an *encoder* consumes the
  whole input into a summary state, and a *decoder* generates an output sequence
  of a different length. Machine translation is the canonical example, and this
  encoder–decoder shape is where attention and sequence-to-sequence models begin.

In every case the recurrence relation and its BPTT are unchanged.

## Bidirectional and deep RNNs

The next two variants stack the same cell rather than change it, so they reuse
the vanilla forward pass and its gradients instead of needing a fresh derivation.

### Bidirectional RNNs

A plain RNN at step $t$ has seen $x^{\langle 1\rangle}\ldots x^{\langle t\rangle}$
but nothing after it. Often the right label depends on what comes next: in "He
said, *Teddy* bears are on sale," *Teddy* is not a name, but in "He said, *Teddy*
Roosevelt was president," it is — and only the following word decides. A
bidirectional RNN (Schuster & Paliwal, 1997) runs two independent recurrences
over the sequence, one left-to-right ($\overrightarrow{a}$) and one
right-to-left ($\overleftarrow{a}$), and concatenates their states at each step:

$$a^{\langle t\rangle} = \big[\,\overrightarrow{a}^{\langle t\rangle};\ \overleftarrow{a}^{\langle t\rangle}\,\big].$$

The output layer then sees both past and future context through a
$2 n_a$-dimensional state. There is no new cell mathematics — it is the same
recurrence run twice, once over the reversed input. The cost is structural: the
backward direction cannot start until the last element has arrived, so a
bidirectional RNN cannot run on a streaming input and is unsuitable for real-time
(online) settings.

### Deep RNNs

Depth is added by stacking. Layer $l$ runs a full recurrence over the sequence of
activations produced by the layer below, consuming $a^{[l-1]\langle t\rangle}$
wherever a single-layer RNN would read $x^{\langle t\rangle}$; only the top
layer's read-out is a model output. Again the cell is unchanged — a deep RNN is
the vanilla forward pass composed once per layer. In practice RNNs are stacked
far more shallowly than feedforward networks, with two or three layers being
common, because the unrolled network is already extremely deep along the time
axis: every extra layer multiplies that cost by the full sequence length while
compounding the same gradient-flow difficulty.

## Where RNNs stand today

The vanilla RNN is the foundation of sequence modelling, but as of 2026 it is
rarely the final architecture for a task with genuine long-range structure. Its
vanishing-gradient limit drove two developments that now dominate.

- **Gated cells** — the LSTM and the GRU — keep the recurrent form but add gates
  and a near-linear state path so gradients survive across many steps. They are
  the direct answer to the problem this page ends on, and they are the subject of
  a sibling topic in this family.
- **Attention and the Transformer** (Vaswani et al., 2017) removed the sequential
  recurrence altogether, relating any two positions in a single step and training
  in parallel across the whole sequence. Transformers now underpin most
  large-scale language and sequence models, and a separate topic covers how they
  relate to — and where they still lose to — recurrent models.

None of this retires the vanilla RNN as an idea. The recurrence, the unrolled
graph, and BPTT are the vocabulary in which gated cells and even "why attention
won" are explained, so understanding this page is what makes the rest of the
family legible.

## Implementation Examples

### Complete Implementations:

> #### **[Vanilla RNN (NumPy)](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/nn/sequence/rnn.py)** - From-scratch recurrence, forward propagation through time, and term-by-term BPTT, with global-norm gradient clipping and the bidirectional and deep forward passes built as compositions of the one cell.
> #### **[Vanilla RNN (PyTorch parity port)](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/pytorch/sequence/rnn.py)** - The same cell as a `torch.nn.RNN` with an `nn.Linear` + softmax read-out, whose autograd gradients reproduce the hand-derived BPTT on the shared fixture to a relative error below `1e-6` — the proof that the derivation on this page is the one PyTorch computes. It also documents the three conventions in which `torch.nn.RNN` diverges from Ng's notation: two bias vectors, a separately attached read-out, and the `(T_x, m, features)` shape order.

## Related documentation

This topic is published as a family of views that share one set of equations and
one implementation. This page is the explanation; the other views are listed
below, and each link activates as its view is published.

| Reader wants to | Go to |
| --- | --- |
| Learn it step by step | [Recurrent neural networks (tutorial notebook)](../tutorials/01-recurrent-neural-networks.ipynb) |
| Do it in a project | Not written for this topic — the application (language modelling and text generation) is a separate topic in this family |
| Look up an equation or shape | [Recurrent neural networks (reference)](../reference/recurrent-neural-networks.md) |
| Understand why it works | Recurrent neural networks — this page |
| Learn by running it | [Recurrent neural networks (tutorial notebook)](../tutorials/01-recurrent-neural-networks.ipynb) |

## References

- **Learning representations by back-propagating errors — Rumelhart, Hinton & Williams (1986)**  
  *Nature* 323, 533–536 – the backpropagation algorithm that BPTT unrolls through time.
- **Finding Structure in Time — Elman (1990)**  
  *Cognitive Science* 14(2), 179–211 – the simple recurrent network this page explains.
- **Deep Learning — Goodfellow, Bengio & Courville (2016), Ch. 10**  
  [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/) – reference derivation of forward propagation and BPTT.
- **Learning long-term dependencies with gradient descent is difficult — Bengio, Simard & Frasconi (1994)**  
  *IEEE Transactions on Neural Networks* 5(2), 157–166 – the original vanishing-gradient analysis.
- **On the difficulty of training Recurrent Neural Networks — Pascanu, Mikolov & Bengio (2013)**  
  [https://arxiv.org/abs/1211.5063](https://arxiv.org/abs/1211.5063) – exploding gradients and gradient clipping.
- **Bidirectional Recurrent Neural Networks — Schuster & Paliwal (1997)**  
  *IEEE Transactions on Signal Processing* 45(11), 2673–2681 – the bidirectional construction.
- **Attention Is All You Need — Vaswani et al. (2017)**  
  [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) – where the field moved next; developed in the modern sequence models topic.
- **Deep Learning Specialization, Course 5 (Sequence Models), Week 1 — Andrew Ng (2018)**  
  [https://www.coursera.org/specializations/deep-learning](https://www.coursera.org/specializations/deep-learning) – the source of this family's notation.

## Key Takeaways

1. Recurrence exists to give a network variable-length input, parameter sharing
   across positions, and a memory of the past folded into one fixed-size hidden
   state — three things a feedforward network cannot provide.
2. One small cell,
   $a^{\langle t\rangle} = \tanh(W_{aa}\,a^{\langle t-1\rangle} + W_{ax}\,x^{\langle t\rangle} + b_a)$,
   is reused at every step; unrolling it produces a very deep feedforward graph
   with tied weights.
3. BPTT is backpropagation on that unrolled graph. The gradient at each hidden
   state sums a read-out term and a recurrence term, and the shared weights
   accumulate a gradient contribution from every step.
4. Multiplying the same recurrent Jacobian at every step makes the long-range
   gradient grow or decay geometrically: a below-one scale vanishes, an above-one
   scale explodes. This is structural, not a coding error.
5. Gradient clipping rescales the gradient to cap its norm while preserving its
   direction. It cures exploding gradients only; vanishing gradients require an
   architectural change, which is what gated cells provide.
6. One-to-one, one-to-many, many-to-one, and many-to-many (equal and unequal
   length) are wiring patterns over the same cell — they differ only in which
   steps take input and which emit output.
7. Bidirectional and deep RNNs are compositions of the vanilla cell, not new
   mathematics: the bidirectional form trades the ability to run online for
   future context, and deep RNNs stack shallowly because the network is already
   deep in time.
8. As of 2026, gated cells and Transformers have largely displaced the vanilla
   RNN for long-range tasks, but its recurrence, unrolling, and BPTT remain the
   vocabulary those methods are explained in.
