# How to choose a gated cell

You have a sequence task, a vanilla RNN that will not learn the dependency you
care about, and a decision to make: GRU, LSTM, or one of the 2024 minimal
variants. This guide goes from that symptom to a wired, verified cell — which
cell to pick and why, how many parameters it costs, how to run the hub's
reference implementation, what breaks when you hand the same cell to Keras or
PyTorch, and when the minimal variants are worth the swap.

It assumes you already know the vanilla RNN and backpropagation through time.
The gate equations themselves are stated in lookup form in this topic's
reference view, and the reason gating works is argued in its explanation view;
both are reachable from the [Crosswalk](#crosswalk).

## Contents

- [Check that gating is the fix you need](#check-that-gating-is-the-fix-you-need)
- [Pick GRU or LSTM](#pick-gru-or-lstm)
- [Size the cell](#size-the-cell)
- [Wire up the reference implementation](#wire-up-the-reference-implementation)
- [Prove the wiring before you train](#prove-the-wiring-before-you-train)
- [Move the cell into a framework layer](#move-the-cell-into-a-framework-layer)
- [Reach for minGRU or minLSTM when the sequence is long](#reach-for-mingru-or-minlstm-when-the-sequence-is-long)
- [Diagnose a cell that will not train](#diagnose-a-cell-that-will-not-train)
- [Implementation Examples](#implementation-examples)
- [References](#references)
- [Crosswalk](#crosswalk)
- [Key Takeaways](#key-takeaways)

## Check that gating is the fix you need

A gated cell buys exactly one thing: an **additive, elementwise** path along the
state. For the LSTM, holding the gates fixed,

$$\frac{\partial c^{\langle t\rangle}}{\partial c^{\langle t-1\rangle}} = \Gamma_f^{\langle t\rangle},$$

so carrying gradient back $k$ steps multiplies $k$ gate values rather than $k$
Jacobians. A forget gate near $1$ transports gradient across many steps
essentially undamped, where the vanilla RNN's repeated Jacobian product decays
geometrically (Bengio, Simard & Frasconi, 1994). The GRU's $(1 - \Gamma_u)$ path
plays the same role.

Reach for a gated cell when:

- the dependency you need spans more than a handful of timesteps, and
- the symptom is a **flat** loss on exactly that dependency — short-range
  structure is learned, long-range structure is not.

Do not expect gating to fix these, because it does not address them:

- **Exploding gradients.** Gating answers the vanishing half of the problem
  only. Keep clipping by global norm (Pascanu, Mikolov & Bengio, 2013); see
  `clip_gradients` below.
- **Throughput.** Both classical cells are serial in $T_x$: step $t$ needs
  $a^{\langle t-1\rangle}$. If training time rather than accuracy is your
  constraint, jump to
  [the minimal variants](#reach-for-mingru-or-minlstm-when-the-sequence-is-long).
- **Content-based mixing across the whole sequence.** That is attention, a
  different architecture family, and out of scope for this topic.

## Pick GRU or LSTM

Start from the GRU and move to the LSTM for a reason you can name.

**Choose the GRU when** the parameter budget or per-step cost binds: three gate
matrices instead of four, one state to carry instead of two, and roughly a
quarter fewer cell parameters at equal hidden width.

**Choose the LSTM when** any of these apply:

- You want memory the read-out cannot see. The LSTM keeps $c^{\langle t\rangle}$
  separate from $a^{\langle t\rangle}$ and the output gate decides how much of it
  is exposed at each step; the GRU has one state doing both jobs.
- You are reproducing a published result or a pretrained checkpoint that used an
  LSTM.
- You will cross framework boundaries. Every LSTM in the ecosystem computes the
  same gate equations, while "GRU" names two different functions — see
  [Move the cell into a framework layer](#move-the-cell-into-a-framework-layer).

**Do not expect the choice to decide accuracy.** Chung, Gulcehre, Cho and Bengio
(2014) compared the two on sequence modelling and found both gated units clearly
beat a plain $\tanh$ unit, but reached no conclusive verdict between them. That
result has held up: as of 2026 the tie-break is cost and interoperability, not a
known accuracy gap.

| | GRU | LSTM |
| --- | --- | --- |
| Gates | reset $\Gamma_r$, update $\Gamma_u$ | forget $\Gamma_f$, update $\Gamma_u$, output $\Gamma_o$ |
| States threaded through time | $a^{\langle t\rangle}$ | $a^{\langle t\rangle}$ and $c^{\langle t\rangle}$ |
| Weight matrices | 3 — $W_r, W_u, W_c$ | 4 — $W_f, W_u, W_c, W_o$ |
| Cell parameters | $3\,n_a\,(n_a + n_x + 1)$ | $4\,n_a\,(n_a + n_x + 1)$ |
| Framework reproduces the hub's form | TensorFlow with `reset_after=False` only | yes |

The update gate is written $\Gamma_u$ in both cells, following Ng's Course 5
notation, which is what makes their correspondence visible. Hochreiter and
Schmidhuber's *input* gate $i$ is the same gate as the LSTM's $\Gamma_u$ here.

## Size the cell

Each gate owns one $(n_a,\, n_a + n_x)$ matrix and one $(n_a,\, 1)$ bias, so the
budget is the gate count times $n_a (n_a + n_x + 1)$, plus the softmax read-out:

```python
def gated_parameter_count(n_x: int, n_a: int, n_y: int, cell: str) -> int:
    """Parameters in one gated cell plus its softmax read-out."""
    gates = {"gru": 3, "lstm": 4}[cell]
    # One (n_a, n_a + n_x) matrix and one (n_a, 1) bias per gate.
    recurrent = gates * n_a * (n_a + n_x + 1)
    read_out = n_y * (n_a + 1)  # W_y and b_y
    return recurrent + read_out
```

At $n_x = 128$, $n_a = 256$, $n_y = 10$ that is **298,250** parameters for the
GRU and **396,810** for the LSTM. Two consequences worth planning around:

- The cell is **quadratic in $n_a$** and only linear in $n_x$. Widening the
  hidden state is the expensive move; widening the input embedding is not.
- At a fixed budget you can afford a wider GRU than LSTM. Comparing a GRU and an
  LSTM at equal *parameter count* rather than equal $n_a$ is usually the
  comparison your deployment actually faces.

## Wire up the reference implementation

The hub's cells take one dictionary of parameters and return caches for the
backward pass. Gates are ordered **forget, update, candidate, output** for the
LSTM, and each weight matrix is the **concatenated** form $W_g = [\,W_{ga} \mid W_{gx}\,]$
acting on the stacked $[\,a^{\langle t-1\rangle};\, x^{\langle t\rangle}\,]$.

```python
from dlhub.nn.sequence.lstm import lstm_backward, lstm_forward, make_fixture
from dlhub.nn.sequence.rnn import compute_loss

fixture = make_fixture()
x, a0, c0 = fixture["x"], fixture["a0"], fixture["c0"]
parameters, y = fixture["parameters"], fixture["y"]

# a, c: (n_a, m, T_x); y_pred: (n_y, m, T_x)
a, c, y_pred, caches = lstm_forward(x, parameters, a0, c0)
loss = compute_loss(y_pred, y)

# dWf, dWu, dWc, dWo, dWy, the matching db_*, plus da0, dc0 and dx.
gradients = lstm_backward(y, caches, parameters)
```

Substituting the GRU changes two lines and nothing else: `gru_forward` returns
`(a, y_pred, caches)` — there is no cell state — and its gradients are keyed
`dWr`, `dWu`, `dWc` instead of the LSTM's four. Both cells import `softmax` and
`compute_loss` from `dlhub.nn.sequence.rnn`, so the read-out and loss cannot
drift away from the vanilla RNN's.

Two things to get right when you take the step:

```python
from dlhub.nn.sequence.lstm import PARAMETER_KEYS
from dlhub.nn.sequence.rnn import clip_gradients

learning_rate = 0.01

# `gradients` also holds da0, dc0 and dx, which are not parameters. Select the
# parameter gradients first, or the clipping norm is computed over the wrong set.
parameter_gradients = {f"d{key}": gradients[f"d{key}"] for key in PARAMETER_KEYS}
clipped = clip_gradients(parameter_gradients, max_norm=5.0)

# `update_parameters` in rnn.py is hard-coded to the vanilla RNN's five keys and
# raises KeyError on a gated dictionary. One comprehension replaces it.
parameters = {
    key: parameters[key] - learning_rate * clipped[f"d{key}"] for key in PARAMETER_KEYS
}
```

Initial states default to zero, so pass `a0` and `c0` only when you are
continuing a sequence across batches. To inspect one gate's two halves — the
recurrent path and the input path — split the concatenated matrix at column
$n_a$ with `split_gate_matrix(gradients["dWf"], n_a)`.

## Prove the wiring before you train

A sign error in a gated backward pass trains the model uphill while every shape
assertion still passes, and four gates give it four more places to hide than the
vanilla RNN has. Check the gradients before you spend a GPU hour on them.

1. **Run the module.** `python -m dlhub.nn.sequence.lstm` prints the shapes, the
   loss, and the gradient norms on the shared fixture, with each gate gradient
   split into its recurrent and input halves.
2. **Run the suites.** `pytest tests/test_lstm.py tests/test_gru.py` compares
   every analytic gradient — including `da0`, `dc0` and `dx` — against a central
   finite difference and requires a relative error below `1e-7`. Do this in
   **float64**: float32 is too coarse for the comparison to mean anything. (The
   shipped suites take the finite-difference step at `1e-6`, one decade above the
   helper's default, because a smaller step puts the roundoff floor of a
   float64 difference uncomfortably close to the tolerance.)
3. **Keep a test that can fail.** Each suite flips a sign and asserts the metric
   rejects it. A tolerance nothing can violate is not a check.

Two of the shipped tests state the architecture's thesis as an assertion rather
than a claim: that $\partial c^{\langle t\rangle} / \partial c^{\langle t-1\rangle}$
is exactly diagonal with the forget gate on the diagonal, and that an open forget
gate delivers gradient to $c^{\langle 0\rangle}$ that a shut one destroys. If you
re-derive the backward pass yourself, port those two tests first.

## Move the cell into a framework layer

**The trap:** the reset gate has two placements in circulation, and the layer is
called `GRU` in both cases. This hub is canonically the Cho/Ng form, in which the
reset gate is applied **before** the hidden transform, inside the concatenation
that $W_c$ multiplies. PyTorch computes
$n_t = \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{t-1} + b_{hn}))$ instead,
applying the reset **after** the recurrent transform — which lets all three
recurrent products be taken in one batched matrix multiply, and is the
cuDNN-compatible form Keras exposes as `reset_after=True`. The two are not
algebraically equal.

Measured on 2026-09-19 against `tf-nightly 2.22.0.dev20260912` /
`keras-nightly 3.16.0.dev` and `torch 2.14.0+cpu`, on the topic's fixture with
`return_sequences=True`:

| Framework cell | Matches the reference? | Measured max abs error |
| --- | --- | --- |
| `tf.keras.layers.LSTM` | yes | `2.2e-16` |
| `tf.keras.layers.GRU(reset_after=False)` | yes | `2.2e-16` |
| `tf.keras.layers.GRU(reset_after=True)` | **no** — cuDNN reset-after form | `1.79` |
| `torch.nn.GRU` | **no** — reset-after form, and no flag to change it | — |

So: **transplant a GRU only into TensorFlow, and only with `reset_after=False`.**
If you must use `torch.nn.GRU`, adopt its definition and train from scratch
rather than porting weights across. `torch.nn.LSTM` computes the hub's gate
equations with the bias split into `b_ih` and `b_hh` (whose sum is the single
$b_g$ here), so the LSTM has no equivalent divergence — but only the TensorFlow
side has been measured here, so verify before you trust a PyTorch transplant.

To move hub parameters into a Keras GRU, transpose them (Keras computes
`inputs @ kernel`), split each gate matrix at column $n_a$, and pack the gates in
Keras' order — **GRU `[z, r, h]`**, **LSTM `[i, f, c, o]`**. One sign flip is
required and is easy to miss: Keras writes the blend as
$h = z \odot h_{t-1} + (1 - z) \odot \tilde{h}$, putting $z$ on the **old** state,
so its $z$ is this hub's $1 - \Gamma_u$. Both are sigmoids, so the flip is a
negated pre-activation rather than a different gate.

```python
import numpy as np

from dlhub.nn.sequence import split_gate_matrix


def keras_gru_weights(parameters: dict[str, np.ndarray], n_a: int) -> list[np.ndarray]:
    """Pack hub GRU parameters for `tf.keras.layers.GRU(reset_after=False)`."""
    packed = [
        (-parameters["Wu"], -parameters["bu"]),  # z, the negated update gate
        (parameters["Wr"], parameters["br"]),  # r
        (parameters["Wc"], parameters["bc"]),  # h, the candidate
    ]
    kernel, recurrent, bias = [], [], []
    for W_g, b_g in packed:
        W_ga, W_gx = split_gate_matrix(W_g, n_a)  # recurrent half | input half
        kernel.append(W_gx.T)  # Keras computes inputs @ kernel
        recurrent.append(W_ga.T)
        bias.append(b_g.ravel())
    return [np.hstack(kernel), np.hstack(recurrent), np.concatenate(bias)]
```

`reset_after=False` gives a single bias row, so `kernel` is $(n_x,\, 3 n_a)$,
`recurrent_kernel` is $(n_a,\, 3 n_a)$, and `bias` is $(3 n_a,)$. The LSTM uses
the same packing with `4`$n_a$ throughout, the order `(Wu, Wf, Wc, Wo)`
— hub $\Gamma_u$ is Keras' input gate `i` — and **no** negation anywhere.

```python
import numpy as np
import tensorflow as tf

tf.keras.backend.set_floatx("float64")  # float32 cannot carry the tolerance

layer = tf.keras.layers.GRU(
    n_a,
    activation="tanh",
    recurrent_activation="sigmoid",
    reset_after=False,  # the one flag that decides whether this matches
    return_sequences=True,
)
layer.build((None, T_x, n_x))
layer.set_weights(keras_gru_weights(parameters, n_a))

# Keras is batch-first: (m, T_x, n_x) in, (m, T_x, n_a) out.
a_keras = layer(np.transpose(x, (1, 2, 0)), initial_state=[a0.T]).numpy()
assert np.max(np.abs(np.transpose(a_keras, (2, 0, 1)) - a)) < 1e-8
```

Check **gradients**, not only the forward pass: take them with `GradientTape`
from the identical fixture arrays in float64 and require a relative error below
`1e-6` against the NumPy BPTT. If you flipped the sign on the wrong gate, the
forward pass can still look plausible while the gradient does not. The
corresponding gradient carries the same flip, `dWu = -dWz`.

## Reach for minGRU or minLSTM when the sequence is long

minGRU and minLSTM (Feng et al., 2024) delete one thing from the classical
cells: **the gates stop depending on $a^{\langle t-1\rangle}$** and read
$x^{\langle t\rangle}$ alone. The recurrence becomes first-order linear,
$a^{\langle t\rangle} = \alpha^{\langle t\rangle} \odot a^{\langle t-1\rangle} + \beta^{\langle t\rangle}$,
whose coefficients are computable for every timestep at once — so an associative
(parallel) scan replaces the serial loop.

**Take the swap when** $T_x$ runs to hundreds or thousands of steps, training
throughput is your binding constraint, and you are happy to let autodiff produce
the backward pass. **Do not take it when** you need the per-step expressivity of
a gate that has read the state so far — that dependence is precisely what was
removed.

What goes away, and what a reader arriving from the classical cells will be
tempted to restore:

- minGRU has **no reset gate**: the candidate no longer reads the previous
  state, so there is nothing left to reset.
- minLSTM has **no output gate**, and with $a^{\langle t\rangle} = c^{\langle t\rangle}$
  the separate cell state collapses too. Its two remaining gates are normalised
  to sum to one, which gives the state a time-independent scale.
- **The candidate is a plain linear map — there is no $\tanh$.** This is a
  deliberate part of the reduction, not an omission to fix.

Parameter counts fall accordingly: minGRU is $O(2 d_h d_x)$ against the GRU's
$O(3 d_h (d_x + d_h))$, and minLSTM $O(3 d_h d_x)$ against the LSTM's
$O(4 d_h (d_x + d_h))$.

```python
import numpy as np

from dlhub.nn.sequence.min_gated import (
    g,
    make_fixture,
    min_gru_forward_log,
    min_gru_forward_parallel,
    min_gru_forward_sequential,
)

fixture = make_fixture()
x, a0, parameters = fixture["x"], fixture["a0"], fixture["min_gru"]

serial = min_gru_forward_sequential(x, parameters, a0)  # the definition
scanned = min_gru_forward_parallel(x, parameters, a0)  # same function, reassociated
assert np.max(np.abs(serial - scanned)) < 1e-10

# The log-space scan is a different *cell*: its candidate is g, not the
# identity, and it needs a non-negative initial state.
positive_a0 = np.abs(a0)
log_space = min_gru_forward_log(x, parameters, positive_a0)
same_cell = min_gru_forward_sequential(x, parameters, positive_a0, candidate=g)
assert np.max(np.abs(log_space - same_cell)) < 1e-10
```

Two practical cautions:

- **Test your scan at an odd length.** The fixture uses $T_x = 7$ on purpose: a
  scan that quietly assumes a power-of-two length passes at $T_x = 4$ and fails
  here.
- **Use the log-space form for long sequences.** Every $\alpha$ lies strictly
  inside $(0, 1)$, so the prefix product decays exponentially and underflows over
  a few hundred steps of a nearly-closed gate — sooner in float32. The log-space
  scan keeps each product as a sum of logs, at the price of a round trip through
  `log` and `exp` (hence the `1e-10` equivalence tolerance rather than machine
  epsilon).

These cells were published in 2024 and, as of 2026, are a promising
simplification rather than a settled default. Measure them against a classical
cell on your own task before committing.

## Diagnose a cell that will not train

| Symptom | First thing to check |
| --- | --- |
| Loss diverges or goes NaN | Clip by global norm. Gating fixes vanishing, not exploding gradients (Pascanu, Mikolov & Bengio, 2013). |
| Loss falls then flattens; long-range dependency never learned | Inspect the forget gate. Initialising $b_f$ to a large value such as $1$ starts the cell out remembering (Goodfellow, Bengio & Courville, 2016, Ch. 10). |
| Gradient check fails for one gate only | Split that gate's matrix at column $n_a$ and compare the halves — the recurrent and input paths fail differently. |
| Forward pass matches a framework but gradients do not | You flipped a sign on the wrong gate; the gradient carries the same flip as the weight, `dWu = -dWz`. |
| Framework output differs by order $0.1$ or more, not by rounding | The GRU reset placement. Set `reset_after=False`, or accept the other definition. |
| Gradient check passes in float32 | It cannot. Re-run the check in float64; train in float32. |
| Minimal-cell state underflows to zero over long sequences | Switch to the log-space scan, and give it a non-negative initial state. |

## Implementation Examples

### Complete Implementations:

> #### **[LSTM — from scratch (NumPy)](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/nn/sequence/lstm.py)** - The three gates, the separate cell state, forward propagation through time, and the hand-derived BPTT, gradient-checked term by term against central finite differences on the shared fixture.
> #### **[GRU — from scratch (NumPy)](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/nn/sequence/gru.py)** - The two-gate cell in the Cho/Ng form, with the reset gate applied before the hidden transform, and the same gradient-checked BPTT. Its module notes document the reset-placement divergence this guide warns about.
> #### **[minGRU and minLSTM — from scratch (NumPy)](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/nn/sequence/min_gated.py)** - The 2024 minimal variants in all three forms — sequential recurrence, plain-space parallel scan, and the stable log-space scan — checked against each other and against the classical cells they reduce from.
> #### **[LSTM and GRU — TensorFlow parity port](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/tensorflow/sequence/lstm_gru.py)** - How to build these cells with `tf.keras.layers.LSTM` and `tf.keras.layers.GRU` and get the equations above rather than something near them. It names the arguments that matter — `reset_after=False` for the Cho/Ng GRU, `unit_forget_bias` for the LSTM's initialization — and shows the weight mapping, including the sign flip Keras's inverted update gate forces. Checked against the NumPy references on the shared fixtures to `atol=1e-8` forward and below `1e-15` relative error on every gradient.

Rendered signatures and docstrings for the three NumPy modules are in
the generated
[neural-network API reference](../reference/api/nn.md).

## References

- **Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. _Neural Computation_ 9(8), 1735–1780.** – The original LSTM, and the source of the input gate this hub writes $\Gamma_u$.
- **Gers, F. A., Schmidhuber, J. & Cummins, F. (2000). Learning to Forget: Continual Prediction with LSTM. _Neural Computation_ 12(10), 2451–2471.** – The forget gate, without which the 1997 cell cannot release memory.
- **Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H. & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. EMNLP.**  
  [https://arxiv.org/abs/1406.1078](https://arxiv.org/abs/1406.1078) – The GRU, in the reset-before-transform form this hub follows.
- **Chung, J., Gulcehre, C., Cho, K. & Bengio, Y. (2014). Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling.**  
  [https://arxiv.org/abs/1412.3555](https://arxiv.org/abs/1412.3555) – The GRU-versus-LSTM comparison behind this guide's "no conclusive winner".
- **Feng, L., Tung, F., Ahmed, M. O., Bengio, Y. & Hajimirsadeghi, H. (2024). Were RNNs All We Needed?**  
  [https://arxiv.org/abs/2410.01201](https://arxiv.org/abs/2410.01201) – minGRU and minLSTM, the parallel scan, and the log-space implementation.
- **Bengio, Y., Simard, P. & Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult. _IEEE Transactions on Neural Networks_ 5(2), 157–166.** – The vanishing-gradient analysis gating answers.
- **Pascanu, R., Mikolov, T. & Bengio, Y. (2013). On the difficulty of training Recurrent Neural Networks. ICML.**  
  [https://arxiv.org/abs/1211.5063](https://arxiv.org/abs/1211.5063) – Exploding gradients and global-norm clipping, which gating does not replace.
- **Goodfellow, I., Bengio, Y. & Courville, A. (2016). _Deep Learning_, Ch. 10.**  
  [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/) – Gated architectures, including the forget-gate bias initialisation.
- **Ng, A. (2018). Deep Learning Specialization, Course 5 (Sequence Models), Week 1.**  
  [https://www.coursera.org/specializations/deep-learning](https://www.coursera.org/specializations/deep-learning) – The notation used here, including $\Gamma_u$ for both cells' update gate.
- **Keras `GRU` layer documentation.**  
  [https://keras.io/api/layers/recurrent_layers/gru/](https://keras.io/api/layers/recurrent_layers/gru/) – The `reset_after` flag and its cuDNN-compatible default.
- **PyTorch `torch.nn.GRU` documentation.**  
  [https://pytorch.org/docs/stable/generated/torch.nn.GRU.html](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html) – The reset-after recurrence, with no option to change it.

## Crosswalk

Live links are added as each sibling ships; entries marked *planned* are not yet
published.

| Reader wants to | Go to |
| --- | --- |
| Learn it step by step | [LSTM and GRU (notebook)](../tutorials/02-lstm-and-gru.ipynb) |
| Do it in a project | Choose a gated cell — *this page* |
| Look up an equation or shape | [LSTM and GRU](../reference/lstm-and-gru.md) |
| Understand why it works | [LSTM and GRU](../explanation/lstm-and-gru.md) |
| Learn by running it | [LSTM and GRU (notebook)](../tutorials/02-lstm-and-gru.ipynb) |

## Key Takeaways

1. Gating replaces the vanilla RNN's repeated Jacobian product with an
   elementwise factor — $\partial c^{\langle t\rangle} / \partial c^{\langle t-1\rangle} = \Gamma_f^{\langle t\rangle}$
   for the LSTM — which is what carries gradient across many steps. It does
   nothing about exploding gradients, so keep clipping.
2. Default to the GRU on cost, and move to the LSTM for a nameable reason:
   separate memory behind an output gate, a checkpoint to reproduce, or a
   framework boundary to cross. Chung et al. (2014) found no accuracy verdict
   between them, and none has arrived since.
3. Budget $3\,n_a(n_a + n_x + 1)$ parameters for a GRU and
   $4\,n_a(n_a + n_x + 1)$ for an LSTM, plus the read-out. The cell is quadratic
   in $n_a$, so hidden width is the expensive knob.
4. Gradient-check before training, in float64, against central finite
   differences, to a relative error below `1e-7` — and keep a test that a
   flipped sign actually fails.
5. "GRU" names two different functions. The hub, Cho and Ng apply the reset gate
   **before** the hidden transform; `torch.nn.GRU` and
   `tf.keras.layers.GRU(reset_after=True)` apply it after. Only
   `reset_after=False` reproduces this hub, measured to `2.2e-16` against `1.79`
   for the default.
6. Porting a GRU to Keras needs a transpose, a split at column $n_a$, the gate
   order `[z, r, h]`, and one sign flip, because Keras puts its $z$ on the old
   state: $z = 1 - \Gamma_u$, so $W_z \leftarrow -W_u$ and $dW_u = -dW_z$.
7. minGRU and minLSTM drop the gates' dependence on $a^{\langle t-1\rangle}$,
   which makes the recurrence first-order linear and parallel-scannable. They
   also drop the reset gate, the output gate, and the candidate's $\tanh$ — all
   deliberately. Test any scan at a length that is odd and not a power of two.
