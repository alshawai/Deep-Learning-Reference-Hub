# LSTM and GRU

A vanilla RNN cannot learn a dependency that spans many timesteps, and no
amount of tuning fixes it: the obstruction is in the architecture, not in the
hyperparameters. The LSTM and the GRU are the two gated cells that answer that
problem, and they answer it in the same way — by giving the state a path along
which it is *carried* rather than *recomputed*, and letting the network learn,
per unit and per timestep, when to use it.

This page argues why that works, what it costs, what it does not fix, and how
the two cells relate to each other and to the 2024 minimal variants that strip
them back. It assumes you already know the vanilla RNN, backpropagation through
time, and the vanishing-gradient analysis — all of which are argued in
[Recurrent neural networks](recurrent-neural-networks.md), the sibling this page
builds on. Notation follows Andrew Ng's Deep Learning Specialization, Course 5:
hidden states $a^{\langle t\rangle}$, inputs $x^{\langle t\rangle}$, the LSTM's
cell state $c^{\langle t\rangle}$, and gates $\Gamma_{\cdot}$. Each gate owns one
matrix acting on the stacked vector
$[\,a^{\langle t-1\rangle};\, x^{\langle t\rangle}\,]$.

For the equations and tensor shapes in lookup form, for the decision procedure
that picks a cell, and for a runnable version, see the
[Crosswalk](#crosswalk).

## Contents

- [The problem the gates were invented for](#the-problem-the-gates-were-invented-for)
- [What a gate is](#what-a-gate-is)
- [The LSTM: a state you write to rather than recompute](#the-lstm-a-state-you-write-to-rather-than-recompute)
- [The GRU: one gate doing two jobs](#the-gru-one-gate-doing-two-jobs)
- [Why the carry path transports gradient](#why-the-carry-path-transports-gradient)
- [Two cells, one idea](#two-cells-one-idea)
- [The same cell name, two different functions](#the-same-cell-name-two-different-functions)
- [What happens when the gates forget the state](#what-happens-when-the-gates-forget-the-state)
- [Where gated cells stand today](#where-gated-cells-stand-today)
- [Implementation Examples](#implementation-examples)
- [References](#references)
- [Crosswalk](#crosswalk)
- [Key Takeaways](#key-takeaways)

## The problem the gates were invented for

In a vanilla RNN every hidden state is produced by pushing the previous one
through a weight matrix and a squashing nonlinearity, so the gradient that
reaches $k$ steps into the past is a product of $k$ copies of one Jacobian,

$$\frac{\partial a^{\langle t\rangle}}{\partial a^{\langle t-k\rangle}} = \prod_{i=t-k+1}^{t} \mathrm{diag}\!\big(1 - (a^{\langle i\rangle})^2\big)\, W_{aa},$$

whose magnitude is governed geometrically by the largest singular value of the
repeated factor (Bengio, Simard & Frasconi, 1994; Pascanu, Mikolov & Bengio,
2013). The two ways that goes wrong are not symmetric. Exploding gradients are a
*scaling* problem, and scaling has a scaling fix: clip by global norm and the
direction survives. Vanishing gradients are an *information* problem. Once the
signal from step $t-50$ has decayed into the rounding error at step $t$, no
rescaling brings it back, because there is nothing left to rescale.

So the architecture has to change, and the thing that has to change is specific:
the **default behaviour of the state**. In a vanilla RNN, keeping a value for
fifty steps means fifty consecutive multiply-and-squash operations that happen to
reconstruct it. Nothing in the parameterisation makes "keep what you had" either
cheap or reliably reachable — remembering is as much work as computing, and the
work is done by the same weights that have to serve every other step too.

Hochreiter and Schmidhuber (1997) started from the opposite default. Build a unit
whose state persists unchanged unless something intervenes — their *constant
error carousel*, a self-loop with weight exactly one, along which error neither
grows nor shrinks — and then attach learned, differentiable valves that decide
when to intervene. Everything else about the LSTM follows from that inversion:
the cell state is the carousel, and the gates are the valves.

## What a gate is

A gate is a vector of numbers in $(0, 1)$, computed from the same two things the
cell reads, and used to scale another vector elementwise:

$$\Gamma^{\langle t\rangle} = \sigma\!\big(W\,[\,a^{\langle t-1\rangle};\, x^{\langle t\rangle}\,] + b\big), \qquad \text{applied as } \Gamma^{\langle t\rangle} \odot v.$$

Four properties of that definition each do real work.

- **Bounded in $(0, 1)$.** The gate is a soft switch: near $0$ it blocks, near
  $1$ it passes, in between it attenuates. Because the bound is two-sided it is
  an interpolation weight, which is what lets two gated terms be *added* without
  the sum running away.
- **Differentiable.** An `if` statement would settle the same question and could
  not be trained. The sigmoid's derivative comes for free from the cached gate,
  $\sigma'(z) = \Gamma(1 - \Gamma)$, so the backward pass evaluates no new
  nonlinearity — the same trick the vanilla RNN uses for its single $\tanh$.
- **Elementwise.** One gate value per hidden unit, so each unit gets its own
  timescale: some can hold a value for the whole sequence while their neighbours
  turn over every step. It also has a mechanical consequence that matters more
  than it looks. Multiplying elementwise keeps the carry Jacobian **diagonal** —
  unit $i$'s memory reaches unit $i$ and no other — where the vanilla RNN's
  $W_{aa}$ mixes every unit into every other at every step.
- **Data-dependent.** The gate is recomputed at each step from the current input
  and the current state, so the cell's timescale is chosen per example and per
  position rather than fixed at design time. A gate is not a hyperparameter; it
  is an activation.

The cost of a gate is a full weight matrix the size of the vanilla RNN's entire
recurrence, plus its bias. That is the price list for the rest of this page: the
GRU pays it three times, the LSTM four.

## The LSTM: a state you write to rather than recompute

The LSTM threads two states through time. For $t = 1 \ldots T_x$, in the order
this hub computes them — forget, update, candidate, output:

$$\Gamma_f^{\langle t\rangle} = \sigma\!\big(W_f\,[\,a^{\langle t-1\rangle};\,x^{\langle t\rangle}\,] + b_f\big), \qquad \Gamma_u^{\langle t\rangle} = \sigma\!\big(W_u\,[\,a^{\langle t-1\rangle};\,x^{\langle t\rangle}\,] + b_u\big)$$

$$\tilde{c}^{\langle t\rangle} = \tanh\!\big(W_c\,[\,a^{\langle t-1\rangle};\,x^{\langle t\rangle}\,] + b_c\big), \qquad \Gamma_o^{\langle t\rangle} = \sigma\!\big(W_o\,[\,a^{\langle t-1\rangle};\,x^{\langle t\rangle}\,] + b_o\big)$$

$$c^{\langle t\rangle} = \Gamma_u^{\langle t\rangle} \odot \tilde{c}^{\langle t\rangle} + \Gamma_f^{\langle t\rangle} \odot c^{\langle t-1\rangle}, \qquad a^{\langle t\rangle} = \Gamma_o^{\langle t\rangle} \odot \tanh\!\big(c^{\langle t\rangle}\big)$$

Read the cell-state line first, because it is the architecture. The new cell
state is an **addition** of two gated terms: some of what was already there, plus
some of a proposal. Nothing is pushed through a weight matrix on the way from
$c^{\langle t-1\rangle}$ to $c^{\langle t\rangle}$. The three gates are then
three separate decisions about that one line.

- **Forget, $\Gamma_f$** — how much of the old cell state survives. The 1997 cell
  had no such gate: its carousel had a fixed self-weight of one, so a unit could
  be written but never released, and a state that latched onto a feature carried
  it for the rest of the sequence (and grew without bound on long streams). Gers,
  Schmidhuber and Cummins (2000) added the forget gate, and it is the cell
  everyone now calls an LSTM.
- **Update, $\Gamma_u$** — how much of the candidate is written. This is
  Hochreiter and Schmidhuber's *input* gate $i$; Ng writes it $\Gamma_u$ and this
  hub follows, because the same symbol makes its correspondence with the GRU's
  update gate visible at a glance.
- **Output, $\Gamma_o$** — how much of the cell state is *exposed* as the hidden
  state. It gates what leaves the cell, not what the cell holds.

Two structural consequences of that split are worth naming, because they are the
LSTM's real differences from the GRU.

**Memory the read-out cannot see.** Since
$a^{\langle t\rangle} = \Gamma_o^{\langle t\rangle} \odot \tanh(c^{\langle t\rangle})$,
the cell can hold a value it is not currently acting on. Everything downstream —
the softmax read-out, and the next step's own gates — sees only what the output
gate exposes. A unit can therefore keep a fact recorded for later use without
that fact perturbing the intermediate predictions, which is a thing the GRU,
whose entire state is its output, cannot do.

**The cell state is not squashed.** $c^{\langle t\rangle}$ is a running sum whose
magnitude is limited only by how often the update gate opens, so it can pass
outside $(-1, 1)$ and genuinely *integrate* evidence over time. The squashing
moves to the read-out, where $\tanh(c^{\langle t\rangle})$ keeps
$a^{\langle t\rangle}$ bounded as before. That freedom is real capacity, and it
has a matching failure mode: a cell whose forget gate stays open can drift to a
large $c$, where $\tanh(c)$ saturates and the hidden state stops responding to
further change.

## The GRU: one gate doing two jobs

Cho et al. (2014) reached a cell with the same carry property and one gate fewer.
For the GRU the hidden state *is* the cell state,
$a^{\langle t\rangle} = c^{\langle t\rangle}$:

$$\Gamma_r^{\langle t\rangle} = \sigma\!\big(W_r\,[\,a^{\langle t-1\rangle};\,x^{\langle t\rangle}\,] + b_r\big), \qquad \Gamma_u^{\langle t\rangle} = \sigma\!\big(W_u\,[\,a^{\langle t-1\rangle};\,x^{\langle t\rangle}\,] + b_u\big)$$

$$\tilde{a}^{\langle t\rangle} = \tanh\!\big(W_c\,[\,\Gamma_r^{\langle t\rangle} \odot a^{\langle t-1\rangle};\,x^{\langle t\rangle}\,] + b_c\big)$$

$$a^{\langle t\rangle} = \Gamma_u^{\langle t\rangle} \odot \tilde{a}^{\langle t\rangle} + \big(1 - \Gamma_u^{\langle t\rangle}\big) \odot a^{\langle t-1\rangle}$$

Three things distinguish this from the LSTM, and all three come from the same
simplification.

**The blend is convex.** The two coefficients are $\Gamma_u$ and $1 - \Gamma_u$,
which are positive and sum to one, so the new state is a weighted average of
where the cell was and what it proposes. An average cannot leave the range of the
things averaged, so the GRU's state stays bounded by the largest candidate it has
ever seen, with no output squashing required. This is why the cell needs no
output gate to stay stable — and, as the minimal variants below show, it is the
property that has to be preserved when an output gate is removed on purpose.

**Tying forget to write is a genuine restriction.** In the LSTM, $\Gamma_f$ and
$\Gamma_u$ are independent: the cell can keep the old value *and* add to it (both
gates open), or clear a slot without writing anything into it (both shut). The
GRU's single gate makes those the same decision — keeping more necessarily means
writing less, because its forget coefficient is exactly $1 - \Gamma_u$. Whether
that restriction costs accuracy is precisely the question the empirical
literature has never settled (see [Two cells, one idea](#two-cells-one-idea)).

**The reset gate is a different kind of gate.** $\Gamma_r$ does not act on what
the cell keeps or emits; it acts on what the *candidate is allowed to read*. With
$\Gamma_r$ near zero the proposal
$\tilde{a}^{\langle t\rangle}$ is computed from $x^{\langle t\rangle}$ alone, so
the cell can propose a fresh start without having first thrown away the state it
is still carrying — Cho et al.'s (2014) motivation was exactly this, letting a
unit drop history that has become irrelevant. Note where the gate sits: inside
the concatenation that $W_c$ multiplies, *before* the hidden transform. That
placement is not a detail, and it is the subject of
[The same cell name, two different functions](#the-same-cell-name-two-different-functions).

## Why the carry path transports gradient

The whole argument for both cells is one partial derivative. For the LSTM,
holding the gates fixed,

$$\frac{\partial c^{\langle t\rangle}}{\partial c^{\langle t-1\rangle}} = \Gamma_f^{\langle t\rangle},$$

and it is worth saying what that is *not*. It is not a matrix product, so units
do not mix and no singular value gets raised to a power. It is not filtered
through a saturating nonlinearity, so it is not capped by a $\tanh$ derivative.
And it is not the same factor at every step, because the gate is recomputed from
the data. Carrying gradient back $k$ steps along this path multiplies $k$
*numbers* the network itself computed, $\prod_i \Gamma_f^{\langle i\rangle}$,
rather than $k$ Jacobians it cannot choose freely.

That last point is the deep one. In a vanilla RNN the per-step gradient factor
and the forward dynamics are the *same* $W_{aa}$: you cannot tune the decay
without changing what the cell computes. In a gated cell the carry factor is a
separate, input-dependent quantity with its own weights, so the cell can hold
$\Gamma_f \approx 1$ on the units that must remember while the rest of its
capacity does other work. The GRU's $1 - \Gamma_u$ coefficient plays the same
role: when the update gate is shut, the state — and its gradient — is copied
forward instead of being pushed through $W_c$ and another $\tanh$.

The hub's implementations turn each of those claims into a measurement rather
than an assertion.

- **The cell-state Jacobian really is the forget gate.** Differencing one LSTM
  step unit by unit gives a Jacobian whose off-diagonal entries are *exactly*
  zero and whose diagonal is $\Gamma_f^{\langle t\rangle}$.
- **An open gate carries, a shut gate destroys.** Driving the forget gate open
  ($\Gamma_f = 0.9997$) instead of shut ($\Gamma_f = 0.00034$) on the topic's
  fixture multiplies the gradient norm arriving at $c^{\langle 0\rangle}$ by
  about $6{,}700$ — over four timesteps.
- **The GRU's carry is undamped, not merely large.** With the update gate shut,
  the gradient at $a^{\langle 0\rangle}$ comes out equal to the plain *sum* of
  every step's read-out gradient, each with weight one. A path that damped at all
  would make those $T_x$ terms shrink geometrically and the equality fail.

### What the clause "holding the gates fixed" hides

That clause is doing work, and an honest explanation has to open it. The gates
are themselves functions of $a^{\langle t-1\rangle}$, so the carry path is not
the only route backwards through a step. In the LSTM, all four gate
pre-activations deposit the recurrent half of $W_g^{\top}\,\mathrm{d}z_g$ into
the gradient at $a^{\langle t-1\rangle}$; in the GRU, three separate paths
deposit gradient there — the blend, the candidate (through $W_c$ and then $\Gamma_r$), and the two
gates. Those extra routes look exactly like a vanilla RNN's: a weight matrix and
a saturating derivative, decaying geometrically when iterated. Gating does not
delete them. It *adds* one route that need not decay.

So the claim to hold is not "gated cells do not have vanishing gradients". It is
this: the cell has one route whose per-step damping is a learned, data-dependent
number the optimizer can drive toward $1$, instead of a fixed spectral property
of a weight matrix that is busy doing something else. The product
$\prod_i \Gamma_f^{\langle i\rangle}$ is still strictly less than one, and still
geometric — a gate held at $0.9$ retains $2.7 \times 10^{-5}$ of the gradient
after 100 steps, while a gate at $0.999$ retains $90\%$. Gating makes long-range
memory **reachable**, not automatic. Which of those two regimes a trained cell
lands in is a fact about the learned parameters, not a guarantee from the
architecture.

Two pieces of standard practice fall straight out of that reading.

- **Initialise the forget-gate bias positive.** At the usual small random
  initialisation the gate sits near $0.5$, so the carry path damps by $2^{-k}$
  before training has learned anything at all — and the gradient that would teach
  it to remember has to survive that same damping to arrive. Setting $b_f$ to a
  value such as $1$ starts the cell out remembering (Goodfellow, Bengio &
  Courville, 2016, Ch. 10).
- **Saturation cuts both ways.** Since $\sigma'(z) = \Gamma(1 - \Gamma)$, a gate
  pinned near $0$ or $1$ receives almost no gradient of its own. The mechanism
  that protects a memory also protects the *decision* to keep it, so a cell that
  has learned to hold something is slow to learn to stop.

### What gating does not fix

- **Exploding gradients.** Gating answers the vanishing half of the problem only.
  The additive path removes the geometric decay; it does nothing about growth
  along the gate routes, and a forget gate near $1$ over a long sequence sums many
  contributions into $c^{\langle 0\rangle}$ rather than damping them. Keep
  clipping by global norm (Pascanu, Mikolov & Bengio, 2013).
- **Sequential cost.** Step $t$ still needs $a^{\langle t-1\rangle}$, so training
  is $T_x$ serial steps with no parallelism along time. That constraint, not
  accuracy, is what the minimal variants below attack.
- **Content-based retrieval.** A fixed-size state is a lossy running summary, and
  gates only decide how fast it is overwritten. Reaching back for one specific
  element from 500 steps ago is what attention does; no gate schedule provides it.

## Two cells, one idea

The two cells are close enough that the differences can be listed exactly. Tie
the LSTM's gates with $\Gamma_f = 1 - \Gamma_u$, open its output gate, and drop
the $\tanh$ on the read-out, and the cell-state line becomes the GRU's convex
blend; what remains is that the GRU adds a reset gate on the candidate's view of
the state, while the LSTM keeps a second state the read-out cannot see. The GRU is
not literally a restricted LSTM — the correspondence needs those two amendments —
but it is unmistakably the same idea with a different gate inventory.

What the pair teaches is that the invariant matters more than the inventory. Both
cells have a carry coefficient that the data controls, and a candidate that is
mixed in rather than substituted for the state; everything else — how many gates,
whether the memory is hidden behind an output gate, whether the candidate may
read the state — is a design choice. Chung, Gulcehre, Cho and Bengio (2014)
compared the two on sequence modelling and found both clearly better than a plain
$\tanh$ unit and no conclusive winner between them. That result has held: as of
2026 the tie-break is cost and interoperability rather than a known accuracy gap,
which is why picking one belongs in a
[how-to guide](../how-to/choose-a-gated-cell.md) and not in a theory page.

## The same cell name, two different functions

The reset gate has two placements in circulation, and the layer is called `GRU`
in both cases. This hub follows Cho et al. (2014) and Ng: the gate is applied
**before** the hidden transform, inside the concatenation $W_c$ multiplies.
`torch.nn.GRU` and `tf.keras.layers.GRU(reset_after=True)` compute
$n_t = \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{t-1} + b_{hn}))$ instead,
applying the reset **after** the recurrent transform.

The reason for the second form is not theoretical but computational. If the reset
gate acts after the recurrent matrix multiply, then all three of the GRU's
recurrent products read the *same* unmodified $h_{t-1}$ and can be taken as one
batched matmul, which is what a fused cuDNN kernel needs. Keras exposes exactly
that trade-off as its `reset_after` flag.

The two forms are not algebraically equal, and the gap is not a rounding
difference. Measured on 2026-09-19 against `tf-nightly 2.22.0.dev20260912` on
this topic's fixture, `reset_after=False` reproduces the hub's GRU to a maximum
absolute error of `2.2e-16`, while `reset_after=True` differs by `1.79`. There is
a second, smaller trap in the same place: Keras writes the blend as
$h = z \odot h_{t-1} + (1 - z) \odot \tilde{h}$, putting its gate on the **old**
state, so its $z$ is this hub's $1 - \Gamma_u$ and the mapping carries a sign
flip.

That is worth more than a footnote, because of what it implies. "GRU" does not
name one function. Two papers reporting GRU results may not have run the same
cell; a weight transplant across frameworks can silently compute something else
while every shape check passes; and the equations in a paper, not the layer name
in its code, are what define the architecture. The
[how-to guide](../how-to/choose-a-gated-cell.md) carries the port procedure and
the full measured table. The LSTM has no equivalent divergence among the cells
measured here: `tf.keras.layers.LSTM` reproduces the hub's cell to `2.2e-16`,
and what differs elsewhere is bias factorisation and gate packing order rather
than the function computed.

## What happens when the gates forget the state

Feng, Tung, Ahmed, Bengio and Hajimirsadeghi (2024) asked how much of the gating
machinery is load-bearing if the goal is to train quickly, and answered with a
single deletion: **the gates stop depending on $a^{\langle t-1\rangle}$** and read
$x^{\langle t\rangle}$ alone. For minGRU that leaves

$$\Gamma_u^{\langle t\rangle} = \sigma\!\big(W_u\,x^{\langle t\rangle}\big), \qquad \tilde{a}^{\langle t\rangle} = W_c\,x^{\langle t\rangle}, \qquad a^{\langle t\rangle} = \big(1 - \Gamma_u^{\langle t\rangle}\big) \odot a^{\langle t-1\rangle} + \Gamma_u^{\langle t\rangle} \odot \tilde{a}^{\langle t\rangle}.$$

### Why one deletion buys parallelism

Once the coefficients depend on the input alone, the recurrence is a first-order
linear one,

$$a^{\langle t\rangle} = \alpha^{\langle t\rangle} \odot a^{\langle t-1\rangle} + \beta^{\langle t\rangle},$$

in which every $\alpha$ and $\beta$ is computable for every timestep at once —
one matrix multiply over the whole sequence, with no loop. Each step is then an
affine map $a \mapsto \alpha \odot a + \beta$, and composing two affine maps
gives another:

$$(\alpha_1, \beta_1) \circ (\alpha_2, \beta_2) = (\alpha_2 \odot \alpha_1,\ \alpha_2 \odot \beta_1 + \beta_2).$$

Composition is associative, so the prefix compositions can be built by doubling
a stride — $\lceil \log_2 T_x \rceil$ rounds of an associative scan instead of
$T_x$ serial steps. The serial dependency that made the classical cells slow was
never the gating; it was the gates' dependence on the state.

### Why the other deletions follow

The minimal cells drop more than that one dependency, and each further deletion
is a consequence rather than an independent choice.

- **minGRU has no reset gate.** The candidate no longer reads
  $a^{\langle t-1\rangle}$, so $\Gamma_r$ would scale a state that is then
  multiplied by zero. The hub's suite pins this as *removable* rather than merely
  unused: two independently drawn reset matrices, one of them scaled a
  hundredfold, produce bit-identical trajectories.
- **minLSTM has no output gate, and no separate cell state.** With the output
  gate gone, $a^{\langle t\rangle} = c^{\langle t\rangle}$ and the second state
  collapses into the first. Its two surviving gates are then normalised to sum to
  one,
  $\Gamma_f' = \Gamma_f / (\Gamma_f + \Gamma_u)$ and
  $\Gamma_u' = \Gamma_u / (\Gamma_f + \Gamma_u)$, which makes the recurrence a
  convex blend again — so the state stays inside the convex hull of
  $a^{\langle 0\rangle}$ and the candidates, at a time-independent scale. That
  normalisation is what stands in for the dropped output gate: it restores by
  construction the boundedness the GRU gets for free and the LSTM bought with
  $\Gamma_o$ and a $\tanh$.
- **The candidate is a plain linear map — there is no $\tanh$.** This is the
  reduction most likely to be "helpfully" corrected back by someone arriving from
  the classical cells, and it must not be: the state is kept bounded by the convex
  blend, not by a squashing function.

### What the reduction costs

The gates can no longer condition on what the cell is holding. minGRU's
$\alpha^{\langle t\rangle}$ is a function of $x^{\langle t\rangle}$ only, so the
cell's forgetting schedule is decided by the current input rather than by the
current memory — "overwrite this slot *because* of what is already in it" is no
longer expressible. That per-step expressivity is exactly what the classical
gates bought, and it is what is being traded for parallel training.

The parameter counts fall with it: minGRU is $O(2 d_h d_x)$ against the GRU's
$O(3 d_h (d_x + d_h))$, and minLSTM $O(3 d_h d_x)$ against the LSTM's
$O(4 d_h (d_x + d_h))$. Note what vanished — the $d_h^2$ term. With no recurrent
half to any weight matrix, cell cost becomes linear rather than quadratic in
hidden width.

### The numerical fine print, and why it is interesting

Resolving the scan needs the product of $\alpha$ across a prefix, and every
$\alpha$ lies strictly inside $(0, 1)$, so that product decays exponentially: over
512 steps of a nearly-closed gate it underflows to zero even in float64. The
paper's stable implementation (§B.1) therefore scans in **log space**, keeping
every product as a sum of logs, where a 512-step decay is an unremarkable
negative number. Two details of that rearrangement are instructive.

First, it needs no new derivation. Because $\sigma(z) = e^{-\mathrm{softplus}(-z)}$,
the log-gates come straight off the pre-activation $k = W x$ as
$\log \Gamma_u = -\mathrm{softplus}(-k)$ and
$\log(1 - \Gamma_u) = -\mathrm{softplus}(k)$; no gate is ever formed and then
logged. Second, logs need positive arguments, so §B.2.1 replaces the candidate's
identity with a map $g$ that is strictly positive everywhere and still linear
above zero. That makes the log-space form a **different cell**, not merely a
different implementation of the same one — which is why the hub's equivalence
tests compare the log-space scan against the sequential recurrence *using the same
candidate*, and not against the linear one.

These cells were published in 2024. As of 2026 they are a promising
simplification with a thin evidence base beside twenty-five years of LSTM
practice, and the honest summary is that they should be measured against a
classical cell on your own task rather than adopted as a default. The idea
underneath them is the durable part: writing a recurrence as
$\alpha \odot a + \beta$, with coefficients that do not read the state, is the
same move that makes the current generation of parallel sequence models trainable
at length.

## Where gated cells stand today

As of 2026 the LSTM and the GRU are no longer the default architecture for
large-scale sequence work — that place belongs to attention-based models, for the
reasons the vanilla RNN page's
[closing section](recurrent-neural-networks.md#where-rnns-stand-today) sets out.
What they remain is the right tool for a recognisable set of situations, and the
vocabulary for a much larger one.

They are still competitive where recurrence's structure is an asset rather than a
liability: streaming or online inference, where a fixed-size state advances one
step at a time and nothing has to be re-read; latency- or memory-constrained
deployment, where the per-step cost does not grow with how much history has been
seen; and small-data regimes, where a strong inductive bias toward locality in
time beats a more general model. And the 2024 minimal variants (Feng et al.,
2024) are evidence that the question is not closed: the gate, understood as a
learned, elementwise, data-dependent decay, is the mechanism that modern
selective-state models inherit, whatever they are called.

## Implementation Examples

### Complete Implementations:

> #### **[LSTM — from scratch (NumPy)](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/nn/sequence/lstm.py)** - The three gates, the separate cell state, forward propagation through time, and the hand-derived BPTT, with the cell-state Jacobian and the open-versus-shut forget-gate experiment quoted above pinned as tests.
> #### **[GRU — from scratch (NumPy)](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/nn/sequence/gru.py)** - The two-gate cell in the Cho/Ng form, with the reset gate applied before the hidden transform, the convex blend, and the undamped-carry measurement, all gradient-checked against central finite differences.
> #### **[minGRU and minLSTM — from scratch (NumPy)](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/nn/sequence/min_gated.py)** - The 2024 minimal variants in all three forms — sequential recurrence, plain-space associative scan, and the stable log-space scan — with the reductions to the classical gates checked as equalities rather than described.
> #### **[LSTM and GRU — TensorFlow parity port](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/tensorflow/sequence/lstm_gru.py)** - The same two cells as `tf.keras.layers.LSTM` and `tf.keras.layers.GRU`, whose `GradientTape` gradients reproduce the hand-derived BPTT on the shared fixtures to a relative error below `1e-15` — the proof that the derivations on this page are the ones TensorFlow computes.

Rendered signatures and docstrings for the three NumPy modules are in
the generated
[neural-network API reference](../reference/api/nn.md).

## Crosswalk

| Reader wants to | Go to |
| --- | --- |
| Learn it step by step | [LSTM and GRU (notebook)](../tutorials/02-lstm-and-gru.ipynb) |
| Do it in a project | [Choose a gated cell](../how-to/choose-a-gated-cell.md) |
| Look up an equation or shape | [LSTM and GRU](../reference/lstm-and-gru.md) |
| Understand why it works | LSTM and GRU — *this page* |
| Learn by running it | [LSTM and GRU (notebook)](../tutorials/02-lstm-and-gru.ipynb) |

## References

- **Long Short-Term Memory — Hochreiter & Schmidhuber (1997)**  
  *Neural Computation* 9(8), 1735–1780 – the original LSTM and the constant error carousel; the source of the input gate this hub writes $\Gamma_u$.
- **Learning to Forget: Continual Prediction with LSTM — Gers, Schmidhuber & Cummins (2000)**  
  *Neural Computation* 12(10), 2451–2471 – the forget gate, without which the 1997 cell could write memory but never release it.
- **Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation — Cho et al. (2014)**  
  [https://arxiv.org/abs/1406.1078](https://arxiv.org/abs/1406.1078) – the GRU, in the reset-before-transform form this hub follows.
- **Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling — Chung, Gulcehre, Cho & Bengio (2014)**  
  [https://arxiv.org/abs/1412.3555](https://arxiv.org/abs/1412.3555) – the GRU-versus-LSTM comparison that found no conclusive winner.
- **Were RNNs All We Needed? — Feng, Tung, Ahmed, Bengio & Hajimirsadeghi (2024)**  
  [https://arxiv.org/abs/2410.01201](https://arxiv.org/abs/2410.01201) – minGRU and minLSTM, the parallel scan, and the log-space implementation of §B.1.
- **Learning long-term dependencies with gradient descent is difficult — Bengio, Simard & Frasconi (1994)**  
  *IEEE Transactions on Neural Networks* 5(2), 157–166 – the vanishing-gradient analysis gating answers.
- **On the difficulty of training Recurrent Neural Networks — Pascanu, Mikolov & Bengio (2013)**  
  [https://arxiv.org/abs/1211.5063](https://arxiv.org/abs/1211.5063) – exploding gradients and global-norm clipping, which gating does not replace.
- **Deep Learning — Goodfellow, Bengio & Courville (2016), Ch. 10**  
  [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/) – gated architectures, including the forget-gate bias initialisation.
- **Deep Learning Specialization, Course 5 (Sequence Models), Week 1 — Andrew Ng (2018)**  
  [https://www.coursera.org/specializations/deep-learning](https://www.coursera.org/specializations/deep-learning) – the notation used here, including $\Gamma_u$ for both cells' update gate.
- **Keras `GRU` layer documentation**  
  [https://keras.io/api/layers/recurrent_layers/gru/](https://keras.io/api/layers/recurrent_layers/gru/) – the `reset_after` flag and its cuDNN-compatible default.
- **PyTorch `torch.nn.GRU` documentation**  
  [https://pytorch.org/docs/stable/generated/torch.nn.GRU.html](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html) – the reset-after recurrence, with no option to change it.

## Key Takeaways

1. Gated cells exist to change the *default* behaviour of a recurrent state.
   The vanilla RNN recomputes its state at every step, so remembering costs as
   much work as computing; the LSTM's cell state is carried forward and the gates
   only decide when to intervene (Hochreiter & Schmidhuber, 1997).
2. A gate is a data-dependent, elementwise vector in $(0, 1)$. Being bounded
   makes it an interpolation weight, being differentiable makes it trainable,
   and being elementwise keeps the carry Jacobian diagonal and gives every unit
   its own timescale.
3. The LSTM's three gates are three decisions about one additive line: how much
   old state survives ($\Gamma_f$), how much candidate is written ($\Gamma_u$),
   and how much of the result is exposed ($\Gamma_o$). The separate cell state
   buys memory the read-out cannot see, and a state that is never squashed and
   can therefore integrate.
4. The GRU ties forgetting to writing with a single $\Gamma_u$, which makes its
   update a convex blend — bounded without an output gate — and adds a reset gate
   controlling what the candidate may read rather than what the cell keeps.
5. The mechanism is one derivative,
   $\partial c^{\langle t\rangle} / \partial c^{\langle t-1\rangle} = \Gamma_f^{\langle t\rangle}$:
   an elementwise factor the network computes, not a Jacobian whose scale is a
   fixed property of a shared weight matrix. On the hub's fixture, opening the
   forget gate rather than shutting it multiplies the gradient reaching
   $c^{\langle 0\rangle}$ by about $6{,}700$ over four steps.
6. "Holding the gates fixed" is a real caveat. The gates read the previous state,
   so vanilla-RNN-like paths still exist and still decay, and
   $\prod_i \Gamma_f^{\langle i\rangle}$ is still geometric. Gating makes long
   memory reachable, not automatic — hence the positive forget-gate bias, and
   hence the fact that saturated gates learn slowly.
7. Gating fixes neither exploding gradients (keep clipping), nor the serial cost
   of $T_x$ steps, nor content-based retrieval from a fixed-size state.
8. "GRU" names two different functions. Applying the reset gate after the
   recurrent transform lets all three recurrent products fuse into one matmul,
   which is why frameworks default to it; on this topic's fixture the two forms
   differ by `1.79` where the matching form agrees to `2.2e-16`.
9. minGRU and minLSTM delete one thing — the gates' dependence on
   $a^{\langle t-1\rangle}$ — which turns the recurrence into a first-order linear
   one resolvable by an associative scan in $\lceil \log_2 T_x \rceil$ rounds.
   The dropped reset gate, output gate, and candidate $\tanh$ all follow from
   that deletion, and what is traded away is gates that can condition on the
   state (Feng et al., 2024).
