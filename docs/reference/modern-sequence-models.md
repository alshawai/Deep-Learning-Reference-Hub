# Modern Sequence Models

This page is the lookup reference for the sequence-model landscape: one comparison
table over six model families and five axes — parallel training, training compute
in the sequence length, per-token inference cost and state size, long-range
dependency handling, and maturity — plus a glossary of the terms every view of
this topic uses in the same sense. It is the modern overlay on the classical
recurrent cells: attention and the 2021–2024 "linear-time" alternatives to it,
situated rather than built.

This page states facts and cites results. It makes no argument for why attention
displaced recurrence or why recurrence came back — that reasoning is the
[explanation](../explanation/modern-sequence-models.md) — and it recommends no
model for your task — that decision procedure is the
[how-to](../how-to/choose-a-sequence-model.md). Both are reachable from the
[Crosswalk](#crosswalk). It is code-free: this topic situates the landscape and
ships no implementation.

Every empirical claim below is attributed to a specific paper, dated, and stated
as that paper's own reported result. Asymptotic and architectural facts are
asserted. The line between the two is drawn explicitly in
[What this comparison does not claim](#what-this-comparison-does-not-claim). The
modern sources were re-verified against arXiv on 2026-09-29.

## Contents

- [Definitions](#definitions)
- [The comparison table](#the-comparison-table)
- [Reading the axes](#reading-the-axes)
- [Reported results, cited and dated](#reported-results-cited-and-dated)
- [What this comparison does not claim](#what-this-comparison-does-not-claim)
- [References](#references)
- [Crosswalk](#crosswalk)
- [Key Takeaways](#key-takeaways)

## Definitions

The terms the comparison turns on, each used in exactly this sense across the
explanation and how-to views so the three documents do not drift.

| Term | Meaning |
| --- | --- |
| **Linear-time** | $O(T)$ compute in the sequence length $T$, set against self-attention's $O(T^2)$ compute and $O(T)$ memory in $T$. |
| **Parallel (associative) scan** | The prefix-scan that evaluates a linear recurrence in parallel over the $T$ axis at training time. It is the mechanism that lets a recurrent model train Transformer-style — parallel over positions — while still running as a step-by-step recurrence at inference. |
| **$O(1)$ recurrent state** | A fixed-size state carried across steps, independent of $T$, so each generated token costs the same. This is what gives a recurrent or SSM model **$O(1)$-per-token inference**. |
| **KV-cache** | The Transformer's cache of past keys and values. It grows $O(T)$ with the generated length, so per-token inference is $O(T)$ — the contrast to the fixed recurrent state above. |
| **LTI vs. selective SSM** | A structured state-space model (S4) is **linear time-invariant**: its parameters are fixed across $t$, so the whole sequence map is a single global convolution. **Mamba** makes the parameters **input-dependent** ("selective"), which breaks the convolution form and is recovered by a hardware-aware parallel scan. Mamba-2's **state-space duality (SSD)** frames attention and selective SSMs as two views of one structured-matrix operation. |
| **Path length** | The number of computation steps signal or gradient traverses between positions $t$ and $t'$. Through a recurrence it is $O(\lvert t - t' \rvert)$ — repeated Jacobian products, the vanishing/exploding regime analysed in the [recurrent neural networks reference](recurrent-neural-networks.md). Through self-attention it is $O(1)$: any position attends directly to any other. |

## The comparison table

Six families down, five axes across. Cells marked with a paper are that paper's
reported result, quoted in full and hedged in
[Reported results](#reported-results-cited-and-dated); asymptotic cells are
asserted.

| Family | A. Parallel over $T$ in training? | B. Training compute in $T$ | C. Inference per token / state | D. Long-range handling | E. Maturity (as of 2026) |
| --- | --- | --- | --- | --- | --- |
| **Classical recurrence** — RNN, LSTM, GRU | No — serial; step $t$ needs the state from $t-1$ | $O(T)$, but not parallelisable over $T$ | $O(1)$ fixed state; $O(1)$ per token | Path length $O(\lvert t - t' \rvert)$; gating (LSTM/GRU) extends the reachable range but does not remove the decay (Bengio et al., 1994) | Mature and well-understood; displaced by attention for most tasks that can afford parallel training (Vaswani et al., 2017) |
| **Transformer** — self-attention | Yes — fully parallel over positions | $O(T^2)$ compute, $O(T)$ memory | KV-cache grows $O(T)$; $O(T)$ per token | Path length $O(1)$ — direct all-to-all attention | The dominant architecture for sequence modelling; battle-tested at scale. Its $O(T^2)$ cost at long context is the pressure the rest of the table responds to |
| **S4** — structured LTI SSM | Yes — LTI, so the sequence map is a global convolution | $O(T \log T)$ via FFT convolution | $O(1)$ recurrent state (runs as a recurrence at inference) | Built for long range; strong on the Long Range Arena (Gu, Goel & Ré, 2022) | Established the state-space revival (ICLR 2022); a research line more than a production default |
| **Selective SSM** — Mamba, Mamba-2 | Yes — hardware-aware parallel scan (selectivity breaks the convolution form) | $O(T)$ | $O(1)$ fixed state; $O(1)$ per token | Input-dependent ("selective") parameters let the state route content over long context | Mamba reports 5× inference throughput and million-length context (Gu & Dao, 2023); Mamba-2 reports 2–8× over Mamba and frames the SSD duality (Dao & Gu, ICML 2024). Quality vs. attention at frontier scale is open |
| **Linear attention** — RWKV, RetNet | Yes — parallel or chunkwise-recurrent training form | $O(T)$ | $O(1)$ fixed state; recurrent inference form | Fixed-size state summarises history, trading exact all-to-all recall for linear cost | RWKV scaled to 14B, reported on par with similarly sized Transformers (Peng et al., 2023); RetNet reports parallel/recurrent/chunkwise forms with $O(1)$ inference (Sun et al., 2023) |
| **Modern minimal recurrence** — minGRU/minLSTM, xLSTM | Yes — minGRU/minLSTM via associative scan; xLSTM's matrix-memory mLSTM is parallelisable, its scalar sLSTM is not | $O(T)$ via parallel scan (minimal cells); mixed for xLSTM | $O(1)$ fixed state; $O(1)$ per token | Minimal cells reach range through the scan; xLSTM adds exponential gating and matrix memory for capacity | minLSTM/minGRU reported competitive with recent architectures including Transformers (Feng et al., 2024); xLSTM reported favourable vs. state-of-the-art Transformers and SSMs (Beck et al., 2024). Both 2024, promising rather than settled |

**Hybrid landmark.** Griffin mixes gated linear recurrences (its Hawk component)
with local attention; De et al. (2024) report it matching Llama-2 on 6× fewer
training tokens. It is named here as the recurrence-plus-attention hybrid point on
the map, not as its own row.

## Reading the axes

- **A. Parallel over $T$ in training.** The dividing line of the modern era.
  Classical recurrence is serial because step $t$ consumes the state from $t-1$;
  everything below the classical row recovers parallel training, either by being a
  convolution (S4), by an associative scan over a linear recurrence
  (Mamba, minGRU/minLSTM, mLSTM), or by a chunkwise-recurrent attention form
  (RWKV, RetNet).
- **B. Training compute in $T$.** $O(T^2)$ is attention's; $O(T)$ is the linear-time
  goal the alternatives share; $O(T \log T)$ is S4's FFT convolution. These are
  asymptotic in $T$ and do not fix the constant, which is why the maturity column
  and the how-to guide both matter.
- **C. Inference per token and state size.** The other half of attention's cost:
  its KV-cache grows $O(T)$, so autoregressive generation is $O(T)$ per token,
  while a recurrent or SSM model carries an $O(1)$ fixed state and generates each
  token at constant cost. This axis, not axis B, is often what decides long-context
  serving.
- **D. Long-range handling.** Attention's $O(1)$ path length is its structural
  advantage; a recurrence's $O(\lvert t - t' \rvert)$ path length is the classical
  limitation. The SSM and linear-attention families aim to keep the $O(1)$ state
  while widening the range a fixed state can actually carry.
- **E. Maturity.** A dated, hedged note, carrying the citation for each empirical
  claim. Read it as "what a specific paper reported by 2024," not as a settled
  ranking. The full quotations are below.

## Reported results, cited and dated

Column E compresses each of these; here they are in full, stated as the source's
own reported result and nothing more. None is repeated here as established fact.

| Claim, as reported | Source |
| --- | --- |
| S4 reports strong results on the Long Range Arena benchmark for long-sequence modelling | Gu, Goel & Ré (2022), ICLR 2022 |
| Mamba reports up to **5× higher inference throughput than Transformers** and strong scaling to **million-length** sequences | Gu & Dao (2023) |
| Mamba-2 reports **2–8× faster** than Mamba, and frames attention and selective SSMs as two views of one structured-matrix operation (state-space duality) | Dao & Gu (2024), ICML 2024 |
| RWKV reports scaling **to 14B parameters, on par with similarly sized Transformers**, while training in parallel and running inference as an RNN with fixed state | Peng et al. (2023) |
| RetNet reports **parallel, recurrent, and chunkwise-recurrent** forms with **$O(1)$ inference** cost | Sun et al. (2023) |
| minLSTM and minGRU report being **competitive with recent architectures, including Transformers**, while training via a fully parallel scan | Feng et al. (2024) |
| xLSTM reports performing **favourably against state-of-the-art Transformers and state-space models** at the sizes tested | Beck et al. (2024) |
| Griffin reports **matching Llama-2 on 6× fewer training tokens** with a gated-linear-recurrence-plus-local-attention hybrid | De et al. (2024) |

## What this comparison does not claim

The boundary between what is asserted and what is merely reported, drawn so the
table cannot be read as more than it says.

**Asserted** — asymptotic and architectural, standard results carried without a
per-paper citation:

- self-attention is $O(T^2)$ compute and $O(T)$ memory in $T$;
- an RNN or SSM carries an $O(1)$ fixed state and infers at $O(1)$ per token,
  where the Transformer's KV-cache grows $O(T)$;
- a parallel (associative) scan evaluates a linear recurrence over the $T$ axis at
  training time;
- self-attention's inter-position path length is $O(1)$; a recurrence's is
  $O(\lvert t - t' \rvert)$.

**Cited, dated, and hedged** — every performance number in the table and in
[Reported results](#reported-results-cited-and-dated) is one paper's reported
result at one point in time, not a settled fact. A fast-moving area invites
overstatement; these entries are deliberately quoted rather than paraphrased into
conclusions.

**Not entailed** — this page does **not** claim that:

- Transformers are obsolete or superseded — they remain the dominant, most
  battle-tested family;
- any recurrent, SSM, or linear-attention model universally beats attention on
  quality;
- the revival is a quality story. It is framed as an **efficiency** story at long
  context — linear-time training and $O(1)$ inference — with the quality question
  stated as **open and evolving**.

## References

- **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł. & Polosukhin, I. (2017). Attention Is All You Need. NeurIPS.**  
  [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) – The architecture that displaced RNNs for most sequence tasks; the $O(1)$ path length and $O(T^2)$ cost this table sets everything else against.
- **Gu, A., Goel, K. & Ré, C. (2022). Efficiently Modeling Long Sequences with Structured State Spaces. ICLR.**  
  [https://arxiv.org/abs/2111.00396](https://arxiv.org/abs/2111.00396) – S4; the structured LTI state-space model and the revival's starting point.
- **Gu, A. & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.**  
  [https://arxiv.org/abs/2312.00752](https://arxiv.org/abs/2312.00752) – Selective (input-dependent) SSMs with a hardware-aware parallel scan; the 5× throughput and million-length claims.
- **Dao, T. & Gu, A. (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. ICML.**  
  [https://arxiv.org/abs/2405.21060](https://arxiv.org/abs/2405.21060) – Mamba-2 and the SSD duality between attention and selective SSMs; the 2–8× claim.
- **Beck, M., Pöppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M., Klambauer, G., Brandstetter, J. & Hochreiter, S. (2024). xLSTM: Extended Long Short-Term Memory.**  
  [https://arxiv.org/abs/2405.04517](https://arxiv.org/abs/2405.04517) – Exponential gating, the scalar-memory sLSTM and the parallelisable matrix-memory mLSTM.
- **Feng, L., Tung, F., Ahmed, M. O., Bengio, Y. & Hajimirsadeghi, H. (2024). Were RNNs All We Needed?**  
  [https://arxiv.org/abs/2410.01201](https://arxiv.org/abs/2410.01201) – minGRU and minLSTM; minimal, fully-parallel-trainable recurrence via an associative scan.
- **Peng, B., Alcaide, E., Anthony, Q., Albalak, A., Arcadinho, S., Biderman, S. et al. (2023). RWKV: Reinventing RNNs for the Transformer Era.**  
  [https://arxiv.org/abs/2305.13048](https://arxiv.org/abs/2305.13048) – A linear-attention model that trains in parallel and infers as an RNN with fixed state.
- **Sun, Y., Dong, L., Huang, S., Ma, S., Xia, Y., Xue, J., Wang, J. & Wei, F. (2023). Retentive Network: A Successor to Transformer for Large Language Models.**  
  [https://arxiv.org/abs/2307.08621](https://arxiv.org/abs/2307.08621) – RetNet; the parallel, recurrent, and chunkwise-recurrent forms with $O(1)$ inference.
- **De, S., Smith, S. L., Fernando, A., Botev, A., Cristian-Muraru, G., Gu, A. et al. (2024). Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models.**  
  [https://arxiv.org/abs/2402.19427](https://arxiv.org/abs/2402.19427) – Hawk (gated linear recurrence) and the Griffin hybrid; the 6×-fewer-tokens claim.
- **Bengio, Y., Simard, P. & Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult. _IEEE Transactions on Neural Networks_ 5(2), 157–166.** – The vanishing/exploding-gradient analysis behind the recurrence's $O(\lvert t - t' \rvert)$ path length.

## Crosswalk

| Reader wants to | Go to |
| --- | --- |
| Learn it step by step | not written for this topic — a conceptual overlay, not a hands-on lesson |
| Do it in a project | [Choose a sequence model](../how-to/choose-a-sequence-model.md) |
| Look up the comparison or a definition | Modern sequence models — *this page* |
| Understand why it works | [Modern sequence models](../explanation/modern-sequence-models.md) |
| Learn by running it | not written for this topic — situates the landscape, ships no implementation |

## Key Takeaways

1. The table reads on five axes: whether training parallelises over $T$, training
   compute in $T$, per-token inference cost and state size, long-range handling,
   and a dated maturity note. Axes A–C are asymptotic and asserted; axis E carries
   the citations.
2. Attention's trade is $O(1)$ path length and fully parallel training bought with
   $O(T^2)$ compute and an $O(T)$ KV-cache at inference. Every other family in the
   table is a different point on that trade.
3. The recurrence revival is a **linear-time** story: S4 as a global convolution
   ($O(T \log T)$), selective SSMs (Mamba/Mamba-2) and minimal recurrence
   (minGRU/minLSTM, xLSTM) via an associative scan ($O(T)$), and linear attention
   (RWKV, RetNet) via chunkwise-recurrent training — all keeping an $O(1)$ inference
   state.
4. The selective step matters: S4 is linear time-invariant and so a convolution,
   while Mamba's input-dependent parameters break that form and are recovered by a
   hardware-aware scan; Mamba-2's SSD duality frames attention and selective SSMs
   as one structured-matrix operation.
5. Every performance number here is one paper's reported result, dated and hedged
   — Mamba's 5×, Mamba-2's 2–8×, RWKV at 14B, Griffin's 6×-fewer-tokens, and the
   "competitive"/"favourable" claims for minGRU/minLSTM and xLSTM. None is repeated
   as settled fact.
6. The topic does not claim attention is obsolete or that any alternative
   universally wins on quality. It frames the revival as efficiency at long
   context, with the quality question open and evolving as of 2026.
