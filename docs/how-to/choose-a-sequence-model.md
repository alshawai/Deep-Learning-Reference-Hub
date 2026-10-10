# How to choose a sequence model

You have a sequence task and a decision one level up from the cell: not GRU
versus LSTM, but whether to reach for **attention** at all, or for one of the
linear-time families that came back for long sequences — a structured
state-space model (S4, Mamba), a linear-attention model (RWKV, RetNet), or a
modern minimal or extended recurrence (minGRU/minLSTM, xLSTM). This guide takes
you from the constraints your task actually imposes — sequence length, inference
budget, the quality you need proven, and your hardware — to a defensible default
and the reason for it.

It decides *architecture family*, not the cell inside a recurrent layer. If your
question is GRU versus LSTM versus a minimal variant, that is a different
altitude, answered in [choose a gated cell](choose-a-gated-cell.md). The
asymptotics this guide leans on — attention's $O(T^2)$ compute, an RNN or SSM's
$O(1)$ state — are tabulated in this topic's reference view, and the reasons the
recurrence revival happened are argued in its explanation view; both are
reachable from the [Crosswalk](#crosswalk).

A word on honesty before you start. As of 2026 this is a fast-moving area, and
the recurrence revival is an **efficiency** story — linear-time training and
fixed-cost inference at long context — not a settled quality verdict. Where a
family's quality is a specific paper's reported result rather than a broad
consensus, this guide says so and dates it. Nothing here claims attention is
obsolete.

## Contents

- [Start from attention as the default](#start-from-attention-as-the-default)
- [Name the constraint that moves you off it](#name-the-constraint-that-moves-you-off-it)
- [Reach for a linear-time family when the sequence is long](#reach-for-a-linear-time-family-when-the-sequence-is-long)
- [Match the family to the constraint](#match-the-family-to-the-constraint)
- [Weigh quality that is proven at scale](#weigh-quality-that-is-proven-at-scale)
- [Decision table](#decision-table)
- [Respect what the revival does and does not prove](#respect-what-the-revival-does-and-does-not-prove)
- [Diagnose a model that does not fit the budget](#diagnose-a-model-that-does-not-fit-the-budget)
- [Key Takeaways](#key-takeaways)
- [References](#references)
- [Crosswalk](#crosswalk)

## Start from attention as the default

Default to a Transformer, and move off it for a reason you can name. The reason
to start there is not fashion; it is two properties that are hard to give up:

- **It trains fully parallel over the sequence.** Self-attention has no
  step-by-step recurrence to unroll, so every position is computed at once and
  there is no backpropagation-through-time bottleneck. That is the property that
  let sequence models scale on modern accelerators (Vaswani et al., 2017).
- **Any position reaches any other in one hop.** The signal path length between
  positions $t$ and $t'$ is $O(1)$ through attention, against $O(|t - t'|)$
  through a recurrence — the same repeated-Jacobian path whose vanishing is
  analysed in the [recurrent neural networks reference](../reference/recurrent-neural-networks.md).
  Long-range structure does not have to survive a chain of multiplications to be
  seen.

You also get the deepest tooling, the most pretrained checkpoints, and the most
reproducible recipes. If your sequences are short-to-moderate and you have no
binding latency or memory constraint, stop here — attention is the choice with
the fewest surprises.

What attention costs, and what pushes you to look further, is entirely about
scale in the sequence length $T$: its compute is $O(T^2)$ and its memory $O(T)$
in $T$, and at inference its **KV-cache grows $O(T)$**, so each new token costs
$O(T)$ to produce. When $T$ is large, both the training bill and the per-token
generation cost climb with it.

## Name the constraint that moves you off it

You leave attention for a *specific* pressure, not a general preference. There
are four, and naming yours decides most of the choice:

- **Sequence length.** $T$ runs to tens of thousands of tokens or more, and the
  $O(T^2)$ training term or the $O(T)$-and-growing KV-cache is the wall you are
  hitting.
- **Inference budget and state size.** You generate autoregressively under a
  latency or memory ceiling — on-device, streaming, or high-throughput serving —
  and a KV-cache that grows with every token is what breaks the budget. A
  recurrent or SSM model carries a **fixed-size $O(1)$ state** instead, so each
  token costs the same regardless of how much came before.
- **Quality you can defend at scale.** You need a result that is known to hold at
  the size you will actually train, from a checkpoint or recipe you can point to.
- **Hardware and ecosystem.** What runs well on your accelerator, and what has a
  maintained implementation you trust, is a real constraint, not a footnote.

If none of these bind, you do not have a reason to leave attention. If one or
more do, read on — the family you want depends on which.

## Reach for a linear-time family when the sequence is long

The families that came back for long sequences share one mechanism: they are
**linear recurrences you can train in parallel**. A parallel (associative) scan
evaluates the recurrence over the whole $T$ axis at once at training time — so
you keep attention's parallel-training property — while the same model runs as a
step-by-step recurrence with fixed state at inference. That is the combination
attention cannot offer: $O(T)$ (or $O(T \log T)$) training in $T$ *and* $O(1)$
per-token generation.

Take a linear-time family when:

- $T$ is large enough that $O(T^2)$ training or the growing KV-cache is your
  binding constraint, **or**
- autoregressive inference under a fixed latency or memory budget is, and a
  constant-size state is what you need.

Do **not** reach for one when your sequences are short, attention already fits
your budget, and you would be trading a battle-tested default for a newer family
to solve a problem you do not have. The revival buys efficiency at long context;
if you are not paying attention's long-context cost, there is nothing to buy
back.

## Match the family to the constraint

Four families answer the long-sequence pressure in different ways. Pick by the
constraint that moved you, and by how much proven quality you need (the next
section).

- **Structured state-space models — S4, then Mamba / Mamba-2.** S4 is a
  *linear time-invariant* SSM: its parameters are fixed across time, so the whole
  sequence map is one global convolution, trainable in $O(T \log T)$ (Gu, Goel &
  Ré, 2022). **Mamba** makes those parameters input-dependent ("selective"),
  which drops the convolution form and is recovered by a hardware-aware parallel
  scan; Gu & Dao (2023) report roughly **5× higher inference throughput than
  Transformers** and modelling out to million-length context. **Mamba-2** reframes
  attention and selective SSMs as two views of one structured-matrix operation
  (state-space duality) and Dao & Gu (ICML 2024) report it **2–8× faster** than
  Mamba. Reach here when long-context modelling with strong reported quality is
  the goal and you can adopt an SSM implementation.

- **Linear-attention models — RWKV, RetNet.** These keep an attention-shaped
  formulation but linearise it so it has a recurrent form with $O(1)$ inference
  state. RWKV trains in parallel and infers as an RNN; Peng et al. (2023) report
  scaling **to 14B parameters, on par with similarly sized Transformers**. RetNet
  offers parallel, recurrent, and chunkwise-recurrent forms of the same map, with
  $O(1)$ inference (Sun et al., 2023). Reach here when you want an
  attention-like model whose inference cost per token is flat.

- **Modern minimal and extended recurrence — minGRU/minLSTM, xLSTM.** minGRU and
  minLSTM strip the classical cells' dependence on the previous hidden state so
  the recurrence becomes first-order linear and parallel-scannable; Feng et al.
  (2024) report them **competitive with recent sequence models, including
  Transformers**. xLSTM adds exponential gating and a parallelisable matrix-memory
  variant; Beck et al. (2024) report it **favourable against state-of-the-art
  Transformers and SSMs**. Reach here when you want the smallest step from the
  gated cells you already know — and note that minGRU/minLSTM as an
  *implementation* lives in [choose a gated cell](choose-a-gated-cell.md), which
  is where to go if your decision is really about the cell.

- **Hybrids — Griffin / Hawk.** Mixing a gated linear recurrence with local
  attention is its own answer; De et al. (2024) report Griffin **matching Llama-2
  on 6× fewer training tokens**. Named here as the hybrid landmark: when neither
  pure attention nor a pure recurrence fits, a hybrid is a real third option.

## Weigh quality that is proven at scale

This is the axis most likely to keep you on attention even when the efficiency
math favours a recurrence. The asymptotic and inference-cost advantages above are
**architectural facts** — they hold by construction. The *quality* claims are
each a specific paper's reported result at that paper's scale and benchmarks,
re-verified against arXiv on 2026-09-29, and they are not the same kind of
statement.

So separate the two questions:

- *Will this family be cheaper at my $T$ and inference budget?* Answerable from
  the asymptotics tabulated in this topic's reference view (reachable from the
  [Crosswalk](#crosswalk)). This is the firm part.
- *Will it be at least as good on my task at my scale?* Answerable only by the
  cited results above — dated, scale-specific, and still evolving — and, when the
  stakes are high, by measuring on your own task before you commit.

If you need a quality guarantee at a scale someone has already demonstrated, and
attention meets your budget, that pull is legitimate and often decisive. If your
binding constraint is efficiency at long context and you can afford to validate
quality yourself, a linear-time family is exactly the trade the revival was built
to offer.

## Decision table

Read it as: the constraint that binds you, the default it points to, and the
thing that would send you elsewhere. Efficiency entries are architectural;
quality entries are the cited, dated results above.

| If your binding constraint is | Default to | Move on when |
| --- | --- | --- |
| Nothing — short-to-moderate $T$, budget met | **Transformer / attention** | you start paying the $O(T^2)$ or KV-cache cost |
| Long $T$, want strong *reported* quality | **Mamba / Mamba-2** (selective SSM) | you cannot adopt an SSM implementation, or need a hybrid's local attention |
| Flat per-token inference, attention-like model | **RWKV or RetNet** (linear attention) | the reported quality at your scale is not enough |
| Smallest step from gated cells you know | **minGRU/minLSTM or xLSTM** | you need cell-level detail — see [choose a gated cell](choose-a-gated-cell.md) |
| Long $T$ *and* some global mixing | **Griffin / Hawk** (hybrid) | a pure family already meets the budget |
| A checkpoint or recipe you must reproduce | **whatever it used** | never, until you have re-validated a swap |

The last row is the quiet one: if you are reproducing a published result or
continuing from a pretrained checkpoint, the architecture is chosen for you, and
this whole guide is a plan for your *next* model, not this one.

## Respect what the revival does and does not prove

Keep the claims straight, because the fast pace of the field makes it easy to
overstate what is settled.

- **These are firm** (architectural, standard): attention is $O(T^2)$ compute and
  $O(T)$ memory in $T$; a recurrent or SSM model carries $O(1)$ state and costs
  $O(1)$ per generated token where attention's KV-cache costs $O(T)$; a parallel
  scan trains a linear recurrence over $T$; attention's inter-position path length
  is $O(1)$.
- **These are reported results** (empirical, each a named paper's own claim at its
  own scale, dated 2023–2024): Mamba's throughput and context length, Mamba-2's
  speedup over Mamba, minGRU/minLSTM's competitiveness, xLSTM's favourable
  comparison, Griffin's token efficiency, RWKV's scaling. Treat them as evidence,
  not guarantees, and check them against your task.
- **These the topic does not claim:** that Transformers are obsolete; that any
  recurrent, SSM, or linear-attention model universally beats attention; or that
  the quality question is closed. It is not, as of 2026 — the honest framing is an
  efficiency win at long context, with quality open and evolving.

## Diagnose a model that does not fit the budget

| Symptom | First thing to check |
| --- | --- |
| Training cost explodes as sequences grow | The $O(T^2)$ attention term. If $T$ is large, a linear-time family trains in $O(T)$ or $O(T \log T)$ instead. |
| Per-token generation slows as context fills | The KV-cache, which grows $O(T)$. A recurrent or SSM model's fixed $O(1)$ state keeps per-token cost flat. |
| A linear-time model underperforms attention on short sequences | You may not have a long-sequence problem. The revival buys long-context efficiency; on short $T$, attention's default advantages still hold. |
| A new family's quality does not match the paper | Reported results are scale- and benchmark-specific and dated 2023–24. Re-measure at your scale before trusting the swap. |
| You cannot find a maintained implementation | Ecosystem is a real constraint. Attention and the classical cells have the deepest tooling; weigh that against the efficiency you would gain. |
| Global content-based mixing is what you actually need | That is attention's defining strength. A pure recurrence trades some of it for efficiency; a hybrid (Griffin/Hawk) keeps a local slice of it. |

## Key Takeaways

1. This is an architecture-family decision, not a cell decision. Default to a
   Transformer and move off it only for a named constraint; GRU-versus-LSTM lives
   one level down in [choose a gated cell](choose-a-gated-cell.md).
2. Attention's advantages are architectural: fully parallel training and $O(1)$
   inter-position path length. Its cost is also architectural: $O(T^2)$ compute,
   $O(T)$ memory, and a KV-cache that grows $O(T)$ so per-token generation is
   $O(T)$.
3. Four constraints move you off attention — long sequence length, a fixed
   inference or memory budget, a need for proven-at-scale quality, and hardware or
   ecosystem. Name yours; it decides most of the choice.
4. The linear-time families share one mechanism: a linear recurrence trained by a
   parallel scan, giving $O(T)$-ish training *and* $O(1)$ per-token inference with
   a fixed-size state. That combination is what attention cannot offer at long
   context.
5. Match the family to the constraint: selective SSMs (Mamba/Mamba-2) for
   long-context with strong reported quality, linear attention (RWKV/RetNet) for
   flat per-token inference, minimal/extended recurrence (minGRU/minLSTM, xLSTM)
   for the smallest step from the classical cells, and Griffin/Hawk when a hybrid
   fits.
6. Keep efficiency and quality separate. The complexity advantages are firm; the
   quality claims are each a specific paper's reported result at its own scale,
   dated 2023–24 and still evolving. Measure on your own task before you commit.
7. The revival is an efficiency story, not a verdict. As of 2026 it does not make
   Transformers obsolete and no family universally beats attention — the win is
   long-context cost, with quality open.

## References

- **Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.**  
  [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) – The Transformer; parallel training and $O(1)$ inter-position path length.
- **Gu, A., Goel, K. & Ré, C. (2022). Efficiently Modeling Long Sequences with Structured State Spaces. ICLR.**  
  [https://arxiv.org/abs/2111.00396](https://arxiv.org/abs/2111.00396) – S4; the structured LTI state-space model behind the revival.
- **Gu, A. & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.**  
  [https://arxiv.org/abs/2312.00752](https://arxiv.org/abs/2312.00752) – Selective SSMs; the ~5× throughput and million-length-context results cited here.
- **Dao, T. & Gu, A. (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. ICML.**  
  [https://arxiv.org/abs/2405.21060](https://arxiv.org/abs/2405.21060) – Mamba-2 and the 2–8× speedup over Mamba.
- **Beck, M. et al. (2024). xLSTM: Extended Long Short-Term Memory.**  
  [https://arxiv.org/abs/2405.04517](https://arxiv.org/abs/2405.04517) – Exponential gating and the parallelisable matrix-memory variant.
- **Feng, L., Tung, F., Ahmed, M. O., Bengio, Y. & Hajimirsadeghi, H. (2024). Were RNNs All We Needed?**  
  [https://arxiv.org/abs/2410.01201](https://arxiv.org/abs/2410.01201) – minGRU/minLSTM; competitiveness with recent models including Transformers.
- **Peng, B. et al. (2023). RWKV: Reinventing RNNs for the Transformer Era.**  
  [https://arxiv.org/abs/2305.13048](https://arxiv.org/abs/2305.13048) – Linear-attention model; parallel training, RNN inference, scaling to 14B.
- **Sun, Y. et al. (2023). Retentive Network: A Successor to Transformer for Large Language Models.**  
  [https://arxiv.org/abs/2307.08621](https://arxiv.org/abs/2307.08621) – RetNet; parallel, recurrent, and chunkwise-recurrent forms with $O(1)$ inference.
- **De, S. et al. (2024). Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models.**  
  [https://arxiv.org/abs/2402.19427](https://arxiv.org/abs/2402.19427) – Hawk and the Griffin hybrid; the 6×-fewer-tokens result.

## Crosswalk

| Reader wants to | Go to |
| --- | --- |
| Learn it step by step | not written for this topic — a conceptual overlay, not a hands-on lesson |
| Do it in a project | Choose a sequence model — *this page* |
| Look up the comparison or a definition | [Modern sequence models](../reference/modern-sequence-models.md) |
| Understand why it works | [Modern sequence models](../explanation/modern-sequence-models.md) |
| Learn by running it | not written for this topic — situates the landscape, ships no implementation |
