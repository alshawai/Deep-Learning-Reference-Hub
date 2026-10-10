# Modern sequence models

Recurrent networks were the default way to model a sequence for most of a decade,
and then, over about two years, they were not. Attention displaced them. Then,
starting around 2021, recurrence came back — not the vanilla RNN, but a family of
linear-time models built to do at long context what attention does expensively.
This page argues *why* each of those moves happened: what problem attention
solved that recurrence could not, what recurrence still buys that attention gives
up, and what the 2021–2024 revival actually proves versus what it merely reports.

It assumes you already know the vanilla RNN, backpropagation through time, and the
vanishing-gradient analysis — the vocabulary this page reasons in — all argued in
[Recurrent neural networks](recurrent-neural-networks.md). It does not build a
Transformer, an SSM, or any of the models it names; those are their own topics,
and this one situates them rather than implementing them. For the same landscape
as a lookup table, and for a decision procedure that picks a family for a task,
see the [Crosswalk](#crosswalk). This page is code-free by design.

A word on honesty first, because the area moves fast. Everything here about
*asymptotics and architecture* is a standard fact that holds by construction.
Everything about *performance* is a specific paper's reported result at that
paper's scale, dated and hedged, never restated as settled truth. The line
between the two is drawn explicitly in
[What the revival proves, and what it does not](#what-the-revival-proves-and-what-it-does-not).
The modern sources were re-verified against arXiv on 2026-09-29.

## Contents

- [The two structural limits of recurrence](#the-two-structural-limits-of-recurrence)
- [Why attention displaced it](#why-attention-displaced-it)
- [What recurrence still buys](#what-recurrence-still-buys)
- [The mechanism the revival shares](#the-mechanism-the-revival-shares)
- [S4: a state space you can convolve](#s4-a-state-space-you-can-convolve)
- [Mamba: making the state space selective](#mamba-making-the-state-space-selective)
- [Linear attention: RWKV and RetNet](#linear-attention-rwkv-and-retnet)
- [Minimal and extended recurrence](#minimal-and-extended-recurrence)
- [Hybrids: keeping a slice of attention](#hybrids-keeping-a-slice-of-attention)
- [What the revival proves, and what it does not](#what-the-revival-proves-and-what-it-does-not)
- [Key Takeaways](#key-takeaways)
- [References](#references)
- [Crosswalk](#crosswalk)

## The two structural limits of recurrence

A recurrent network reads a sequence one step at a time, folding each element
into a hidden state that summarises everything before it. That design has two
limits, and both are architectural rather than a matter of tuning.

The first is about *training*. Step $t$ needs the state from step $t-1$, so the
computation is inherently serial: you cannot evaluate position $t$ until you have
evaluated everything before it. On a modern accelerator, which is fast precisely
because it does many things at once, a computation that must proceed one step at a
time leaves most of the hardware idle. The training cost is $O(T)$ in the sequence
length $T$, but it is $O(T)$ that will not parallelise over $T$.

The second is about *range*. The signal — and the gradient — between positions
$t$ and $t'$ travels through the recurrence, so its path length is
$O(\lvert t - t' \rvert)$: it passes through one Jacobian product per step of
separation. That product is what the vanishing/exploding-gradient analysis of
[Recurrent neural networks](recurrent-neural-networks.md#why-long-range-dependencies-are-hard)
is about; gating (the LSTM, the GRU) widens the range a state can carry but does
not remove the decay (Bengio, Simard & Frasconi, 1994). A dependency that spans
thousands of steps has to survive thousands of multiplications to be learned.

These are the two limits attention was built to remove — and, later, the two
properties the revival tried to recover while keeping recurrence's own
advantages. The rest of this page is that story.

## Why attention displaced it

Self-attention answers both limits at once, and that is the whole reason it took
over most of sequence modelling after 2017 (Vaswani et al., 2017).

It has **no recurrence to unroll**. Every position is computed from every other
position in a single operation, so the whole sequence is processed in parallel at
training time — no step-by-step dependency, no backpropagation-through-time
bottleneck. That is the property that let sequence models scale on accelerators
the way feedforward and convolutional networks already had.

And its **inter-position path length is $O(1)$**. Any position attends directly
to any other in one hop, so a long-range dependency does not have to survive a
chain of Jacobian products to be seen. The vanishing-gradient obstruction that
defines recurrence simply does not arise, because there is no long recurrent path
for the signal to decay along.

Both wins are architectural, and neither is free. Attending every position to
every other is $O(T^2)$ compute and $O(T)$ memory in $T$. And at inference, an
autoregressive Transformer caches the keys and values of every past token — the
**KV-cache** — which grows $O(T)$ with the generated length, so producing each
new token costs $O(T)$. For short-to-moderate sequences these costs are a bargain
for what you get. As $T$ grows into the tens of thousands and beyond, the
quadratic training term and the ever-growing cache become the wall. That wall is
the pressure everything below responds to.

## What recurrence still buys

Here is the asymmetry the revival is built on: the two costs attention pays at
long context are exactly the two things a recurrence never pays.

A recurrent (or state-space) model carries a **fixed-size $O(1)$ state**,
independent of $T$. It does not accumulate a cache. So autoregressive generation
costs the same per token no matter how long the context already is — $O(1)$ per
token against attention's $O(T)$. For streaming, on-device, or high-throughput
serving at long context, that flat per-token cost is decisive, and it is a
property attention structurally cannot have as long as it keeps its all-to-all
formulation.

So the revival did not set out to prove attention wrong. It set out to keep
attention's *training* advantage — parallelism over the sequence — while
recovering recurrence's *inference* advantage — the fixed-size state. The
question the whole 2021–2024 line answers is: can you train in parallel over $T$
and still run as an $O(1)$-per-token recurrence at inference? For a *linear*
recurrence, the answer is yes.

## The mechanism the revival shares

Every family below rests on one idea: a **linear recurrence can be evaluated in
parallel** by a prefix scan.

A general RNN's update is nonlinear in the previous state, which is what forces
it to be serial. But if the recurrence is *linear* — each state is an affine
function of the previous state — then composing two steps is itself an affine map,
and affine maps compose associatively. An **associative (parallel) scan** exploits
exactly that: it evaluates the recurrence over the whole $T$ axis at once, in
$O(\log T)$ parallel depth, instead of $T$ serial steps. So the same model can be
computed two ways — as a parallel scan at training time (keeping attention's
parallelism) and as a step-by-step recurrence with fixed state at inference
(keeping recurrence's $O(1)$ per-token cost). That dual identity is the trick the
whole revival turns on.

What differs between the families is *how* they arrange to be a linear recurrence
you can scan — through a convolution, through input-dependent state-space
parameters, or through a linearised attention. The reference view tabulates all
six families on five axes; this page walks the interesting rows as a story.

## S4: a state space you can convolve

The structured state-space model S4 was the revival's starting point (Gu, Goel &
Ré, 2022). It models the sequence with a continuous linear state-space system —
a linear ODE discretised over the steps — whose parameters are **linear
time-invariant (LTI)**: they do not depend on $t$. That constraint is the whole
point. Because the parameters are fixed across time, the entire sequence-to-
sequence map collapses into a single **global convolution** with one long kernel,
which an FFT evaluates in $O(T \log T)$ — parallel over $T$ at training time. At
inference, the very same system runs as a recurrence carrying an $O(1)$ state.

The LTI structure is what makes long range tractable here: with the right
parameterisation of the state matrix, the convolution kernel can represent
dependencies over very long horizons without the per-step decay a nonlinear
recurrence suffers, which is why S4 posted strong results on long-sequence
benchmarks (Gu, Goel & Ré, 2022). S4 established the state-space line as a real
alternative for long sequences; it remained more a research foundation than a
production default, and its LTI constraint is precisely what the next step relaxes.

## Mamba: making the state space selective

An LTI system applies the *same* dynamics to every input, which means it cannot
decide, based on content, what to keep and what to discard — it processes "the"
and a rare named entity with identical parameters. Mamba's move is to make the
state-space parameters **input-dependent**, or *selective*: at each step the model
computes its state-space coefficients from the current input, so it can route,
gate, and forget content-dependently the way a gated RNN does (Gu & Dao, 2023).

That selectivity is powerful and it breaks the convolution. Once the parameters
vary with the input, the map is no longer time-invariant, so it is no longer one
fixed kernel you can FFT. Mamba recovers parallel training with a **hardware-aware
parallel scan** — the recurrence is still *linear* in the state, so it is still
scannable, and the implementation is written to keep the scan's intermediate state
in fast on-chip memory. Gu & Dao (2023) report up to **5× higher inference
throughput than Transformers** and modelling out to **million-length** sequences;
read those as that paper's reported results, not as a settled ranking.

Mamba-2 then reframes the whole picture. Its **state-space duality (SSD)** shows
that attention and selective SSMs are two views of one structured-matrix
operation — the same computation, expressed either as an attention-style matrix or
as a scan — which both explains why the two families reach comparable quality and
yields a faster algorithm; Dao & Gu (ICML 2024) report Mamba-2 **2–8× faster**
than Mamba. The duality is the conceptual payoff of the revival: attention and the
selective recurrence are not rival species but two coordinates on one map.

## Linear attention: RWKV and RetNet

A second route to the same destination starts from attention and *linearises* it.
Standard attention's $O(T^2)$ cost comes from the softmax coupling every query to
every key. Replace that coupling with a form that factorises, and the attention
computation can be rewritten as a **linear recurrence over a fixed-size state** —
which is exactly the object a scan can train in parallel and a recurrence can run
in $O(1)$ per token.

RWKV builds a model that trains in parallel like a Transformer but **infers as an
RNN** with a fixed state; Peng et al. (2023) report scaling **to 14B parameters,
on par with similarly sized Transformers**. RetNet gives the same map three
interchangeable forms — a **parallel** form for training, a **recurrent** form
for $O(1)$ inference, and a **chunkwise-recurrent** form that trades between them —
so one trained model can be served with a flat per-token cost (Sun et al., 2023).
The trade these make is legible: a fixed-size state summarises history rather than
retaining every token exactly, so they give up attention's exact all-to-all
recall in exchange for linear cost. Whether that trade costs quality on a given
task is the open, task-dependent question — not something the architecture
settles.

## Minimal and extended recurrence

The third route asks the bluntest version of the question: how little do you have
to change the classical cells to make them parallel-trainable? "Were RNNs All We
Needed?" strips the LSTM and GRU of the parts of their gates that depend on the
*previous hidden state*, leaving cells whose recurrence is first-order linear —
and therefore scannable. The resulting **minGRU** and **minLSTM** train by a fully
parallel scan, and Feng et al. (2024) report them **competitive with recent
sequence models, including Transformers**. The lesson is pointed: much of what made
the classical cells slow to train was the nonlinearity in the recurrence, and
removing it recovers parallel training while keeping the gated, fixed-state
character.

xLSTM goes the other direction — *extending* the LSTM rather than minimising it.
It adds **exponential gating** and two memory designs: a scalar-memory sLSTM and a
matrix-memory mLSTM. The mLSTM is parallelisable (its recurrence admits a
parallel form); the sLSTM is not, and is kept for the representational power its
memory mixing adds. Beck et al. (2024) report xLSTM performing **favourably
against state-of-the-art Transformers and state-space models** at the sizes
tested. Both lines are 2024 results — promising, and stated as reported, not
settled. Note the boundary of this topic: minGRU/minLSTM *as an implementation*
belongs to the gated-cell topic; here they are landmarks in the revival narrative,
not a build.

## Hybrids: keeping a slice of attention

A pure recurrence trades away attention's exact all-to-all mixing; a pure
Transformer pays the quadratic cost for it. The hybrid answer keeps a *local*
slice of attention and lets a linear recurrence carry the long range. Griffin
does exactly this — mixing gated linear recurrences (its Hawk component) with
local attention — and De et al. (2024) report it **matching Llama-2 on 6× fewer
training tokens**. It is named here as the third point on the map: when neither a
pure recurrence nor pure attention fits, a hybrid is a real option, not a
compromise of last resort.

## What the revival proves, and what it does not

The field moves fast enough that it is easy to read the revival as more than it
is. Keep three registers of claim separate.

**Firm** — asymptotic and architectural, true by construction and carried without
a per-paper citation:

- self-attention is $O(T^2)$ compute and $O(T)$ memory in $T$;
- a recurrent or SSM model carries an $O(1)$ fixed state and infers at $O(1)$ per
  token, where the Transformer's KV-cache grows $O(T)$;
- a parallel (associative) scan evaluates a *linear* recurrence over the $T$ axis
  at training time;
- self-attention's inter-position path length is $O(1)$; a recurrence's is
  $O(\lvert t - t' \rvert)$.

**Reported** — empirical, each a named paper's own result at its own scale and
benchmarks, dated 2023–2024 and stated as such: Mamba's throughput and context
length (Gu & Dao, 2023), Mamba-2's speedup over Mamba (Dao & Gu, 2024),
minGRU/minLSTM's competitiveness (Feng et al., 2024), xLSTM's favourable
comparison (Beck et al., 2024), Griffin's token efficiency (De et al., 2024), and
RWKV's scaling to 14B (Peng et al., 2023). Treat these as evidence, not
guarantees.

**Not claimed** — this page does *not* say that Transformers are obsolete or
superseded (they remain the dominant, most battle-tested family), that any
recurrent, SSM, or linear-attention model universally beats attention on quality,
or that the quality question is closed. As of 2026 it is not. The honest framing
is an **efficiency** story — linear-time training and $O(1)$ inference at long
context — with the quality verdict open and evolving.

## Key Takeaways

1. Recurrence has two structural limits: it trains serially (step $t$ needs step
   $t-1$), and its inter-position path length is $O(\lvert t - t' \rvert)$, so
   long-range signal must survive a chain of Jacobian products. Both are
   architectural, not tuning problems.
2. Attention displaced recurrence by removing both limits — fully parallel
   training and $O(1)$ inter-position path length — at the cost of $O(T^2)$
   compute, $O(T)$ memory, and a KV-cache that grows $O(T)$ so per-token
   generation is $O(T)$.
3. What recurrence still buys is exactly what attention pays for at long context:
   a fixed-size $O(1)$ state and $O(1)$-per-token inference. Recovering that while
   keeping parallel training is the entire aim of the revival.
4. The shared mechanism is that a *linear* recurrence composes associatively, so a
   parallel scan trains it over $T$ at once while the same model runs as an
   $O(1)$-per-token recurrence at inference.
5. The families are different routes to being a scannable linear recurrence: S4 as
   a global convolution ($O(T \log T)$, LTI); Mamba making the state space
   input-dependent (selective, recovered by a hardware-aware scan) with Mamba-2's
   SSD duality unifying attention and selective SSMs; RWKV/RetNet linearising
   attention; and minGRU/minLSTM and xLSTM reworking the classical gated cells.
6. Keep the registers separate. The complexity and path-length facts are firm; the
   performance numbers — Mamba's 5×, Mamba-2's 2–8×, RWKV at 14B, Griffin's 6×,
   the minGRU/minLSTM and xLSTM comparisons — are each one paper's reported result,
   dated 2023–2024 and hedged.
7. The revival is an efficiency story, not a verdict. As of 2026 it does not make
   Transformers obsolete, and no family universally beats attention on quality —
   the win is long-context cost, with the quality question open and evolving.

## References

- **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł. & Polosukhin, I. (2017). Attention Is All You Need. NeurIPS.**  
  [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) – The architecture that displaced RNNs; the $O(1)$ path length and $O(T^2)$ cost this page's arc turns on.
- **Gu, A., Goel, K. & Ré, C. (2022). Efficiently Modeling Long Sequences with Structured State Spaces. ICLR.**  
  [https://arxiv.org/abs/2111.00396](https://arxiv.org/abs/2111.00396) – S4; the structured LTI state-space model and the revival's starting point.
- **Gu, A. & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.**  
  [https://arxiv.org/abs/2312.00752](https://arxiv.org/abs/2312.00752) – Selective (input-dependent) SSMs with a hardware-aware parallel scan; the 5× throughput and million-length results.
- **Dao, T. & Gu, A. (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. ICML.**  
  [https://arxiv.org/abs/2405.21060](https://arxiv.org/abs/2405.21060) – Mamba-2 and the SSD duality between attention and selective SSMs; the 2–8× result.
- **Beck, M., Pöppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M., Klambauer, G., Brandstetter, J. & Hochreiter, S. (2024). xLSTM: Extended Long Short-Term Memory.**  
  [https://arxiv.org/abs/2405.04517](https://arxiv.org/abs/2405.04517) – Exponential gating, the scalar-memory sLSTM and the parallelisable matrix-memory mLSTM.
- **Feng, L., Tung, F., Ahmed, M. O., Bengio, Y. & Hajimirsadeghi, H. (2024). Were RNNs All We Needed?**  
  [https://arxiv.org/abs/2410.01201](https://arxiv.org/abs/2410.01201) – minGRU and minLSTM; minimal, fully-parallel-trainable recurrence via an associative scan.
- **Peng, B., Alcaide, E., Anthony, Q., Albalak, A., Arcadinho, S., Biderman, S. et al. (2023). RWKV: Reinventing RNNs for the Transformer Era.**  
  [https://arxiv.org/abs/2305.13048](https://arxiv.org/abs/2305.13048) – A linear-attention model that trains in parallel and infers as an RNN with fixed state.
- **Sun, Y., Dong, L., Huang, S., Ma, S., Xia, Y., Xue, J., Wang, J. & Wei, F. (2023). Retentive Network: A Successor to Transformer for Large Language Models.**  
  [https://arxiv.org/abs/2307.08621](https://arxiv.org/abs/2307.08621) – RetNet; the parallel, recurrent, and chunkwise-recurrent forms with $O(1)$ inference.
- **De, S., Smith, S. L., Fernando, A., Botev, A., Cristian-Muraru, G., Gu, A. et al. (2024). Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models.**  
  [https://arxiv.org/abs/2402.19427](https://arxiv.org/abs/2402.19427) – Hawk (gated linear recurrence) and the Griffin hybrid; the 6×-fewer-tokens result.
- **Bengio, Y., Simard, P. & Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult. _IEEE Transactions on Neural Networks_ 5(2), 157–166.** – The vanishing/exploding-gradient analysis behind the recurrence's $O(\lvert t - t' \rvert)$ path length.

## Crosswalk

| Reader wants to | Go to |
| --- | --- |
| Learn it step by step | not written for this topic — a conceptual overlay, not a hands-on lesson |
| Do it in a project | [Choose a sequence model](../how-to/choose-a-sequence-model.md) |
| Look up the comparison or a definition | [Modern sequence models](../reference/modern-sequence-models.md) |
| Understand why it works | Modern sequence models — *this page* |
| Learn by running it | not written for this topic — situates the landscape, ships no implementation |
