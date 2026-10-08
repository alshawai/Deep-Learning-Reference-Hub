# LSTM and GRU

This page is the lookup reference for the gated recurrent cells: the GRU's two
gates, the LSTM's three gates and separate cell state, the per-step
backpropagation-through-time (BPTT) gradient equations for both, the parameter
counts, the 2024 minimal variants minGRU and minLSTM with their parallel scan,
the framework mapping and the places frameworks disagree with the equations
below, and the fixture and tolerances every implementation here is held to.

Notation extends the
[recurrent neural networks reference](recurrent-neural-networks.md) verbatim —
same $a^{\langle t\rangle}$, same feature-first shape convention, same read-out
and loss — and follows Andrew Ng's Deep Learning Specialization, Course 5,
Week 1 (Ng, 2018), so the LSTM's input gate is written $\Gamma_u$. This page
states the equations and makes no argument for them, and it recommends nothing:
the reasoning is in this topic's explanation view and the decision procedure is
in its how-to guide, both reachable from the [Crosswalk](#crosswalk).

## Contents

- [Notation and shapes](#notation-and-shapes)
- [Weight form and gate order](#weight-form-and-gate-order)
- [GRU forward equations](#gru-forward-equations)
- [LSTM forward equations](#lstm-forward-equations)
- [Read-out, loss, and initial states](#read-out-loss-and-initial-states)
- [Backpropagation through time](#backpropagation-through-time)
- [LSTM gradient equations](#lstm-gradient-equations)
- [GRU gradient equations](#gru-gradient-equations)
- [Returned gradients](#returned-gradients)
- [The carry path across one step](#the-carry-path-across-one-step)
- [Parameter counts](#parameter-counts)
- [minGRU and minLSTM equations](#mingru-and-minlstm-equations)
- [Scan forms and coefficients](#scan-forms-and-coefficients)
- [The log space scan](#the-log-space-scan)
- [Framework divergences](#framework-divergences)
- [TensorFlow parity mapping](#tensorflow-parity-mapping)
- [Fixture and tolerances](#fixture-and-tolerances)
- [Failure modes](#failure-modes)
- [Implementation Examples](#implementation-examples)
- [References](#references)
- [Crosswalk](#crosswalk)
- [Key Takeaways](#key-takeaways)

## Notation and shapes

| Symbol | Meaning |
| --- | --- |
| $n_x$ | input dimension (features per timestep) |
| $n_a$ | hidden-state dimension |
| $n_y$ | output dimension (classes) |
| $m$ | batch size |
| $T_x$ | number of timesteps |
| $\Gamma_r^{\langle t\rangle}$ | reset gate (GRU only) — how much of the previous state the candidate reads |
| $\Gamma_u^{\langle t\rangle}$ | update gate (both cells); the LSTM's *input* gate $i$ of Hochreiter & Schmidhuber (1997) |
| $\Gamma_f^{\langle t\rangle}$ | forget gate (LSTM only), introduced by Gers, Schmidhuber & Cummins (2000) |
| $\Gamma_o^{\langle t\rangle}$ | output gate (LSTM only) |
| $\tilde{a}^{\langle t\rangle}$, $\tilde{c}^{\langle t\rangle}$ | candidate state (GRU, LSTM) |
| $c^{\langle t\rangle}$ | cell state; separate from $a^{\langle t\rangle}$ in the LSTM, and $a^{\langle t\rangle} = c^{\langle t\rangle}$ in the GRU |
| $\odot$ | elementwise (Hadamard) product |
| $\sigma$ | logistic sigmoid, so every gate lies in $(0, 1)$ |

Every array carries the feature axis first and time last, so one timestep is a
plain 2-D slice.

| Symbol | Shape |
| --- | --- |
| $x$ | $(n_x, m, T_x)$ |
| $a$, hidden states | $(n_a, m, T_x)$ |
| $c$, cell states (LSTM only) | $(n_a, m, T_x)$ |
| $\hat{y}$ | $(n_y, m, T_x)$ |
| $a^{\langle 0\rangle}$, $c^{\langle 0\rangle}$ | $(n_a, m)$ |
| $W_f$, $W_u$, $W_c$, $W_o$, $W_r$ | $(n_a,\, n_a + n_x)$ |
| $b_f$, $b_u$, $b_c$, $b_o$, $b_r$ | $(n_a, 1)$ |
| $W_y$ | $(n_y, n_a)$ |
| $b_y$ | $(n_y, 1)$ |
| $[\,a^{\langle t-1\rangle};\,x^{\langle t\rangle}\,]$ | $(n_a + n_x,\, m)$ |
| $dx$ | $(n_x, m, T_x)$ |

All weights and biases are **shared across every timestep**: there is one copy,
reused at each step, so BPTT accumulates one gradient per name over the whole
sequence.

## Weight form and gate order

Both classical cells use the **concatenated** form: one weight matrix per gate,
acting on the stacked previous state and current input.

$$W_g = [\,W_{ga} \mid W_{gx}\,], \qquad W_g \in \mathbb{R}^{n_a \times (n_a + n_x)}$$

The two halves are never needed separately by the forward or backward pass —
that is the point of the concatenated form — but they are worth naming when
reporting a gradient, because $dW_{ga}$ is the recurrent path and $dW_{gx}$ the
input path. Splitting at column $n_a$ recovers them: $W_{ga}$ is $(n_a, n_a)$
and $W_{gx}$ is $(n_a, n_x)$. (The vanilla RNN reference keeps its two matrices
separate instead, to expose those two paths directly.)

**Gate order** throughout this page, the implementations, and their printed
output is **forget, update, candidate, output** ($f, u, c, o$). Frameworks pack
their gates in their own orders; see
[TensorFlow parity mapping](#tensorflow-parity-mapping).

## GRU forward equations

Cho et al. (2014), in Ng's notation. For the GRU the hidden state and the cell
state are the same object, $a^{\langle t\rangle} = c^{\langle t\rangle}$, so
there is no second state to thread. For $t = 1, \ldots, T_x$:

$$\Gamma_r^{\langle t\rangle} = \sigma\!\big(W_r\,[\,a^{\langle t-1\rangle};\,x^{\langle t\rangle}\,] + b_r\big)$$

$$\Gamma_u^{\langle t\rangle} = \sigma\!\big(W_u\,[\,a^{\langle t-1\rangle};\,x^{\langle t\rangle}\,] + b_u\big)$$

$$\tilde{a}^{\langle t\rangle} = \tanh\!\big(W_c\,[\,\Gamma_r^{\langle t\rangle} \odot a^{\langle t-1\rangle};\,x^{\langle t\rangle}\,] + b_c\big)$$

$$a^{\langle t\rangle} = \Gamma_u^{\langle t\rangle} \odot \tilde{a}^{\langle t\rangle} + \big(1 - \Gamma_u^{\langle t\rangle}\big) \odot a^{\langle t-1\rangle}$$

Two properties follow directly and are worth recording, because tests pin them:

- The state update is a **convex combination** of $\tilde{a}^{\langle t\rangle}$
  and $a^{\langle t-1\rangle}$, so each entry of $a^{\langle t\rangle}$ lies
  between them.
- **The reset gate is applied before the hidden transform**, inside the
  concatenation that $W_c$ multiplies. This is the Cho/Ng form and it is
  canonical for this hub. It is not algebraically equal to the
  reset-after-transform form several frameworks implement; see
  [Framework divergences](#framework-divergences).

## LSTM forward equations

Hochreiter & Schmidhuber (1997) with the forget gate of Gers, Schmidhuber &
Cummins (2000). The LSTM threads a **separate** cell state
$c^{\langle t\rangle}$ alongside $a^{\langle t\rangle}$. For
$t = 1, \ldots, T_x$:

$$\Gamma_f^{\langle t\rangle} = \sigma\!\big(W_f\,[\,a^{\langle t-1\rangle};\,x^{\langle t\rangle}\,] + b_f\big)$$

$$\Gamma_u^{\langle t\rangle} = \sigma\!\big(W_u\,[\,a^{\langle t-1\rangle};\,x^{\langle t\rangle}\,] + b_u\big)$$

$$\tilde{c}^{\langle t\rangle} = \tanh\!\big(W_c\,[\,a^{\langle t-1\rangle};\,x^{\langle t\rangle}\,] + b_c\big)$$

$$\Gamma_o^{\langle t\rangle} = \sigma\!\big(W_o\,[\,a^{\langle t-1\rangle};\,x^{\langle t\rangle}\,] + b_o\big)$$

$$c^{\langle t\rangle} = \Gamma_u^{\langle t\rangle} \odot \tilde{c}^{\langle t\rangle} + \Gamma_f^{\langle t\rangle} \odot c^{\langle t-1\rangle}$$

$$a^{\langle t\rangle} = \Gamma_o^{\langle t\rangle} \odot \tanh\!\big(c^{\langle t\rangle}\big)$$

Unlike the GRU, no gate is applied inside the concatenation: all four gate
matrices multiply the same $[\,a^{\langle t-1\rangle};\,x^{\langle t\rangle}\,]$.
The cell state is built by **addition** of two gated terms, and nothing
multiplies $c^{\langle t-1\rangle}$ but an elementwise gate — see
[The carry path across one step](#the-carry-path-across-one-step).

## Read-out, loss, and initial states

Identical to the vanilla RNN, so the three sequence modules agree:

$$\hat{y}^{\langle t\rangle} = \mathrm{softmax}\!\big(W_y\,a^{\langle t\rangle} + b_y\big), \qquad L = \sum_{t=1}^{T_x} L^{\langle t\rangle}$$

with each per-step term the batch-mean softmax cross-entropy against the
one-hot target,

$$L^{\langle t\rangle} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{n_y} y^{\langle t\rangle}_{k,i} \, \log \hat{y}^{\langle t\rangle}_{k,i}.$$

| Quantity | Default | Note |
| --- | --- | --- |
| $a^{\langle 0\rangle}$ | $\mathbf{0}$, shape $(n_a, m)$ | supply it only when continuing a sequence across batches |
| $c^{\langle 0\rangle}$ | $\mathbf{0}$, shape $(n_a, m)$ | LSTM only |
| $\mathrm{softmax}$, loss | one shared helper, re-exported by the vanilla RNN module | one canonical implementation, including its max-subtraction stability shift |
| gate nonlinearity | the shared overflow-free $\sigma$ | never exponentiates a positive number, so no clipping and no epsilon |

Both cells return the whole state trajectory and the per-timestep caches the
backward pass consumes:

```python
# GRU: one state threaded through time.
a, y_pred, caches = gru_forward(x, parameters, a0)

# LSTM: two states, hidden and cell.
a, c, y_pred, caches = lstm_forward(x, parameters, a0, c0)
```

## Backpropagation through time

BPTT walks the unrolled graph from $t = T_x$ back to $t = 1$. The GRU threads
**one** gradient backward along the recurrence; the LSTM threads **two**, one
along the hidden state and one along the cell state. Both are zero at the final
timestep, which has no successor.

The read-out path is the same as the vanilla RNN's, with sums taken over the
batch axis:

$$dz_y^{\langle t\rangle} = \frac{1}{m}\big(\hat{y}^{\langle t\rangle} - y^{\langle t\rangle}\big), \qquad dW_y^{\langle t\rangle} = dz_y^{\langle t\rangle}\,\big(a^{\langle t\rangle}\big)^{\!\top}, \qquad db_y^{\langle t\rangle} = \sum_{\text{batch}} dz_y^{\langle t\rangle}$$

The gradient arriving at $a^{\langle t\rangle}$ merges **two sources**, the
read-out at step $t$ and the recurrence carried back from step $t+1$:

$$da^{\langle t\rangle} = W_y^{\!\top}\, dz_y^{\langle t\rangle} \;+\; \big(da^{\langle t\rangle}\big)_{\text{from } t+1}$$

Both nonlinearities are differentiated by reading the cached forward value, so
the backward pass evaluates no new sigmoid and no new $\tanh$:

$$\sigma'(z) = \Gamma \odot (1 - \Gamma), \qquad \tanh'(z) = 1 - \tanh^2(z)$$

Because every weight is shared across time, each parameter gradient is the
**sum of its per-step contributions**, $dW_g = \sum_{t=1}^{T_x} dW_g^{\langle t\rangle}$,
and likewise for every bias.

## LSTM gradient equations

Per timestep, in the order the backward pass evaluates them. Write
$da^{\langle t\rangle}$ for the merged hidden gradient above and
$\big(dc^{\langle t\rangle}\big)_{\text{from } t+1}$ for the cell gradient
arriving from the future.

$$d\Gamma_o^{\langle t\rangle} = da^{\langle t\rangle} \odot \tanh\!\big(c^{\langle t\rangle}\big)$$

$$dc^{\langle t\rangle} = \big(dc^{\langle t\rangle}\big)_{\text{from } t+1} + da^{\langle t\rangle} \odot \Gamma_o^{\langle t\rangle} \odot \big(1 - \tanh^2\!\big(c^{\langle t\rangle}\big)\big)$$

The cell-state line is an addition of two products, so each factor's gradient
is the other factor:

$$d\Gamma_u^{\langle t\rangle} = dc^{\langle t\rangle} \odot \tilde{c}^{\langle t\rangle}, \qquad d\tilde{c}^{\langle t\rangle} = dc^{\langle t\rangle} \odot \Gamma_u^{\langle t\rangle}$$

$$d\Gamma_f^{\langle t\rangle} = dc^{\langle t\rangle} \odot c^{\langle t-1\rangle}, \qquad \big(dc^{\langle t-1\rangle}\big)_{\text{from } t} = dc^{\langle t\rangle} \odot \Gamma_f^{\langle t\rangle}$$

Pre-activations, in the topic's gate order:

$$dz_f = d\Gamma_f \odot \Gamma_f \odot (1 - \Gamma_f), \qquad dz_u = d\Gamma_u \odot \Gamma_u \odot (1 - \Gamma_u)$$

$$dz_c = d\tilde{c} \odot \big(1 - (\tilde{c})^2\big), \qquad dz_o = d\Gamma_o \odot \Gamma_o \odot (1 - \Gamma_o)$$

All four gates read the same stacked vector, so each gate weight's gradient is
its pre-activation gradient times that vector, and their contributions back
through the concatenation **add**:

$$dW_g^{\langle t\rangle} = dz_g^{\langle t\rangle}\,\big[\,a^{\langle t-1\rangle};\,x^{\langle t\rangle}\,\big]^{\!\top}, \qquad db_g^{\langle t\rangle} = \sum_{\text{batch}} dz_g^{\langle t\rangle}, \qquad g \in \{f, u, c, o\}$$

$$d\big[\,a^{\langle t-1\rangle};\,x^{\langle t\rangle}\,\big] = W_f^{\!\top} dz_f + W_u^{\!\top} dz_u + W_c^{\!\top} dz_c + W_o^{\!\top} dz_o$$

Its top $n_a$ rows are $\big(da^{\langle t-1\rangle}\big)_{\text{from } t}$ and
its remaining $n_x$ rows are $dx^{\langle t\rangle}$.

## GRU gradient equations

The GRU's update is a convex combination, so differentiating it in its three
arguments gives:

$$d\Gamma_u^{\langle t\rangle} = da^{\langle t\rangle} \odot \big(\tilde{a}^{\langle t\rangle} - a^{\langle t-1\rangle}\big), \qquad d\tilde{a}^{\langle t\rangle} = da^{\langle t\rangle} \odot \Gamma_u^{\langle t\rangle}$$

**Three separate paths** then deposit gradient into $a^{\langle t-1\rangle}$,
and they are summed. Path 1 is the blend itself:

$$\big(da^{\langle t-1\rangle}\big)_{\text{blend}} = da^{\langle t\rangle} \odot \big(1 - \Gamma_u^{\langle t\rangle}\big)$$

Path 2 runs through the candidate. Note that $W_c$ multiplied the **reset**
concatenation, so its weight gradient uses that vector and not the plain one:

$$dz_c = d\tilde{a} \odot \big(1 - (\tilde{a})^2\big), \qquad dW_c^{\langle t\rangle} = dz_c\,\big[\,\Gamma_r^{\langle t\rangle} \odot a^{\langle t-1\rangle};\,x^{\langle t\rangle}\,\big]^{\!\top}, \qquad db_c^{\langle t\rangle} = \sum_{\text{batch}} dz_c$$

Writing $s$ for the top $n_a$ rows of $W_c^{\!\top} dz_c$ — the gradient with
respect to the reset state $\Gamma_r^{\langle t\rangle} \odot a^{\langle t-1\rangle}$
— the product rule splits it between the gate and the state:

$$d\Gamma_r^{\langle t\rangle} = s \odot a^{\langle t-1\rangle}, \qquad \big(da^{\langle t-1\rangle}\big)_{\text{candidate}} = s \odot \Gamma_r^{\langle t\rangle}$$

Path 3 runs through the two gates, which read the plain concatenation:

$$dz_r = d\Gamma_r \odot \Gamma_r \odot (1 - \Gamma_r), \qquad dz_u = d\Gamma_u \odot \Gamma_u \odot (1 - \Gamma_u)$$

$$dW_g^{\langle t\rangle} = dz_g\,\big[\,a^{\langle t-1\rangle};\,x^{\langle t\rangle}\,\big]^{\!\top}, \qquad db_g^{\langle t\rangle} = \sum_{\text{batch}} dz_g, \qquad g \in \{r, u\}$$

$$\big(da^{\langle t-1\rangle}\big)_{\text{gates}} = \big(W_r^{\!\top} dz_r + W_u^{\!\top} dz_u\big)_{[\,:\,n_a\,]}$$

The input gradient collects the bottom $n_x$ rows of all three products:

$$dx^{\langle t\rangle} = \big(W_c^{\!\top} dz_c + W_r^{\!\top} dz_r + W_u^{\!\top} dz_u\big)_{[\,n_a:\,]}$$

## Returned gradients

After the loop, the recurrent gradients have reached the initial states.

| Cell | Parameter gradients | Also returned |
| --- | --- | --- |
| LSTM | $dW_f, dW_u, dW_c, dW_o$ each $(n_a,\, n_a + n_x)$; $db_f, db_u, db_c, db_o$ each $(n_a, 1)$; $dW_y$ $(n_y, n_a)$; $db_y$ $(n_y, 1)$ | $da_0$ and $dc_0$, each $(n_a, m)$; $dx$ $(n_x, m, T_x)$ |
| GRU | $dW_r, dW_u, dW_c$ each $(n_a,\, n_a + n_x)$; $db_r, db_u, db_c$ each $(n_a, 1)$; $dW_y$ $(n_y, n_a)$; $db_y$ $(n_y, 1)$ | $da_0$ $(n_a, m)$; $dx$ $(n_x, m, T_x)$ |
| minGRU, minLSTM | none — no backward pass is derived for the minimal cells; autodiff or a framework supplies it | — |

Every gradient dictionary is keyed by the parameter name with a `d` prefix, so
`dWf` holds $dW_f$. The dictionary also carries the non-parameter entries in the
right-hand column, which is worth knowing before computing a clipping norm over
`.values()`.

## The carry path across one step

The single partial derivative the architecture exists to produce. For the LSTM,
holding the gates fixed,

$$\frac{\partial c^{\langle t\rangle}}{\partial c^{\langle t-1\rangle}} = \Gamma_f^{\langle t\rangle},$$

an **elementwise** factor: the Jacobian is
$\operatorname{diag}\!\big(\Gamma_f^{\langle t\rangle}\big)$. Carrying gradient
back $k$ steps therefore multiplies $k$ gate values rather than $k$ full
Jacobians, and a forget gate near $1$ transports it essentially undamped. The
vanilla RNN's corresponding product is
$\prod \operatorname{diag}\!\big(1 - (a)^2\big) W_{aa}$, bounded by
$\lVert W_{aa}\rVert_2^{\,k}$ and so geometrically decaying when that norm is
below $1$ (Bengio, Simard & Frasconi, 1994).

For the GRU the blend path plays the same role, but only that path:

$$\bigg(\frac{\partial a^{\langle t\rangle}}{\partial a^{\langle t-1\rangle}}\bigg)_{\text{blend}} = \operatorname{diag}\!\big(1 - \Gamma_u^{\langle t\rangle}\big)$$

The candidate and gate paths of the same derivative do carry weight matrices and
a $\tanh$ derivative, so the GRU's total one-step Jacobian is not diagonal. The
undamped term is the one that survives when the update gate is near zero, in
which case the state — and its gradient — is copied forward.

Gating addresses the vanishing half of the problem only. Exploding gradients
remain a separate failure mode with a separate remedy, global-norm clipping
(Pascanu, Mikolov & Bengio, 2013), documented in the vanilla RNN reference.

## Parameter counts

Each classical gate owns one $(n_a,\, n_a + n_x)$ matrix and one $(n_a, 1)$
bias, so a cell costs its gate count times $n_a (n_a + n_x + 1)$. The minimal
cells have no biases and no recurrent half: every weight is $(n_a, n_x)$.

| Cell | Gates | Cell parameters | On the fixture ($n_x = 3$, $n_a = 5$) |
| --- | --- | --- | --- |
| GRU | 3 | $3\,n_a\,(n_a + n_x + 1)$ | 135 |
| LSTM | 4 | $4\,n_a\,(n_a + n_x + 1)$ | 180 |
| minGRU | 1 (plus the candidate map) | $2\,n_a\,n_x$ | 30 |
| minLSTM | 2 (plus the candidate map) | $3\,n_a\,n_x$ | 45 |

The softmax read-out adds $n_y\,(n_a + 1)$ on top for the classical cells, which
is 12 on the fixture's $n_y = 2$. The cell is **quadratic in $n_a$** and only
linear in $n_x$. In the asymptotic form Feng et al. (2024) quote, minGRU is
$O(2 d_h d_x)$ against the GRU's $O(3 d_h (d_x + d_h))$, and minLSTM
$O(3 d_h d_x)$ against the LSTM's $O(4 d_h (d_x + d_h))$.

## minGRU and minLSTM equations

Feng et al. (2024); the section numbers below are that paper's. The reduction
that makes these cells parallelisable is a single deletion: **the gates stop
depending on $a^{\langle t-1\rangle}$** and read $x^{\langle t\rangle}$ alone
(Sec. 3.1.1, Sec. 3.2.1).

**minGRU** (Sec. 3.1.3):

$$\Gamma_u^{\langle t\rangle} = \sigma\!\big(W_u\,x^{\langle t\rangle}\big), \qquad \tilde{a}^{\langle t\rangle} = W_c\,x^{\langle t\rangle}, \qquad a^{\langle t\rangle} = \big(1 - \Gamma_u^{\langle t\rangle}\big) \odot a^{\langle t-1\rangle} + \Gamma_u^{\langle t\rangle} \odot \tilde{a}^{\langle t\rangle}$$

**minLSTM** (Sec. 3.2.4):

$$\Gamma_f^{\langle t\rangle} = \sigma\!\big(W_f\,x^{\langle t\rangle}\big), \qquad \Gamma_u^{\langle t\rangle} = \sigma\!\big(W_u\,x^{\langle t\rangle}\big), \qquad \tilde{a}^{\langle t\rangle} = W_c\,x^{\langle t\rangle}$$

$$\Gamma_f'^{\langle t\rangle} = \frac{\Gamma_f^{\langle t\rangle}}{\Gamma_f^{\langle t\rangle} + \Gamma_u^{\langle t\rangle}}, \qquad \Gamma_u'^{\langle t\rangle} = \frac{\Gamma_u^{\langle t\rangle}}{\Gamma_f^{\langle t\rangle} + \Gamma_u^{\langle t\rangle}}, \qquad a^{\langle t\rangle} = \Gamma_f'^{\langle t\rangle} \odot a^{\langle t-1\rangle} + \Gamma_u'^{\langle t\rangle} \odot \tilde{a}^{\langle t\rangle}$$

What the reduction removes, stated explicitly because each absence reads like an
omission to repair:

- minGRU has **no reset gate** (Sec. 3.1.1). The candidate no longer reads the
  previous state, so there is nothing left to reset.
- minLSTM has **no output gate** (Sec. 3.2.3), and because
  $a^{\langle t\rangle} = c^{\langle t\rangle}$ the separate cell state collapses
  with it. Its two remaining gates are normalised to sum to one exactly,
  $\Gamma_f' + \Gamma_u' = 1$, which is what gives the state a time-independent
  scale and keeps it inside the convex hull of the candidates and the initial
  state.
- **The candidate is a plain linear map: there is no $\tanh$** (Sec. 3.1.2).
- Neither cell has bias terms, and neither has a recurrent weight half.

The normalising denominator needs no epsilon: it is the sum of two strictly
positive sigmoids. It underflows only when both pre-activations fall below about
$-745$, which is exactly the regime the log-space form handles by subtraction
instead of division.

| Symbol | Shape |
| --- | --- |
| $W_f$, $W_u$, $W_c$ | $(n_a, n_x)$ |
| $x$ | $(n_x, m, T_x)$ |
| $a^{\langle 0\rangle}$ | $(n_a, m)$ |
| $\alpha$, $\beta$, $a$ | $(n_a, m, T_x)$ |

## Scan forms and coefficients

Both minimal cells are the **same first-order linear recurrence** (Sec. 2.3):

$$a^{\langle t\rangle} = \alpha^{\langle t\rangle} \odot a^{\langle t-1\rangle} + \beta^{\langle t\rangle}$$

| Cell | $\alpha^{\langle t\rangle}$ | $\beta^{\langle t\rangle}$ |
| --- | --- | --- |
| minGRU | $1 - \Gamma_u^{\langle t\rangle}$ | $\Gamma_u^{\langle t\rangle} \odot \tilde{a}^{\langle t\rangle}$ |
| minLSTM | $\Gamma_f'^{\langle t\rangle}$ | $\Gamma_u'^{\langle t\rangle} \odot \tilde{a}^{\langle t\rangle}$ |

Because $\alpha$ and $\beta$ depend on $x$ alone, they are computable for every
timestep at once. Each step is then an affine map
$a \mapsto \alpha \odot a + \beta$, and composing two of them — first
$(\alpha_1, \beta_1)$, then $(\alpha_2, \beta_2)$ — gives another:

$$(\alpha_1, \beta_1) \circ (\alpha_2, \beta_2) = \big(\alpha_2 \odot \alpha_1,\; \alpha_2 \odot \beta_1 + \beta_2\big)$$

Composition is associative, so the prefix compositions are built by doubling the
stride (an inclusive Hillis–Steele scan) in $\lceil \log_2 T_x \rceil$ rounds
instead of $T_x$ serial steps; entry $t$ then holds the single affine map from
$a^{\langle 0\rangle}$ to $a^{\langle t\rangle}$. The stride doubles under
`while stride < T_x`, so **no power-of-two length is assumed** — a scan that
pads silently passes at $T_x = 4$ and fails at the fixture's $T_x = 7$.

Three resolutions ship, and they are not interchangeable in one respect:

```python
# The definition, and the same function reassociated. These agree to rounding.
a_serial = min_gru_forward_sequential(x, parameters, a0)
a_scanned = min_gru_forward_parallel(x, parameters, a0)

# A different cell: the log-space form's candidate is g, not the identity,
# and it requires a non-negative initial state.
a_logspace = min_gru_forward_log(x, parameters, np.abs(a0))
a_same_cell = min_gru_forward_sequential(x, parameters, np.abs(a0), candidate=g)
```

Measured by running the module on its own fixture (2026-09), for both cells: the
plain-space scan agrees with the sequential recurrence to `4.4e-16`, and the
log-space scan with the same-candidate sequential pass to `4.0e-15`. Over 512
saturated steps the module records agreement to about `2e-12`, which is why the
declared tolerance is `1e-10` rather than machine epsilon.

## The log space scan

Resolving the recurrence needs the product of $\alpha$ across a prefix, and every
$\alpha$ lies strictly inside $(0, 1)$, so that product decays exponentially with
the prefix length and underflows over a few hundred steps of a nearly closed
gate — sooner in float32. Sec. B.1 keeps every product as a **sum of logs**
instead. Writing $A_t = \sum_{i \le t} \log \alpha^{\langle i\rangle}$,

$$a^{\langle t\rangle} = \exp\!\Big(A_t + \log \sum_{j \le t} \exp\big(\log \beta^{\langle j\rangle} - A_j\big)\Big),$$

with the initial state entering as the $j = 0$ term: coefficient one, hence
log-coefficient zero and log-injection $\log a^{\langle 0\rangle}$. The inner
cumulative sum is a `logcumsumexp`, built with the same stride doubling and
`logaddexp` in place of affine composition.

The gate logarithms come straight from the pre-activations
$k_g^{\langle t\rangle} = W_g\,x^{\langle t\rangle}$, so no gate is ever formed
and then logged:

$$\log \Gamma_u = -\mathrm{softplus}(-k_u), \qquad \log\big(1 - \Gamma_u\big) = -\mathrm{softplus}(k_u)$$

| Cell | $\log \alpha$ | $\log \beta$ |
| --- | --- | --- |
| minGRU | $-\mathrm{softplus}(k_u)$ | $-\mathrm{softplus}(-k_u) + \log g(k_c)$ |
| minLSTM | $\log \Gamma_f - \log N$ | $\log \Gamma_u - \log N + \log g(k_c)$ |

where $\log \Gamma_f = -\mathrm{softplus}(-k_f)$,
$\log \Gamma_u = -\mathrm{softplus}(-k_u)$, and the normaliser is taken by
`logaddexp` rather than division:
$\log N = \operatorname{logaddexp}(\log \Gamma_f,\, \log \Gamma_u)$.

A logarithm needs a positive argument, so Sec. B.2.1 replaces the candidate's
identity with a strictly positive map that is still slope-1 linear on the
positive half:

$$g(x) = \begin{cases} x + 0.5 & x \ge 0 \\ \sigma(x) & x < 0 \end{cases}, \qquad \log g(x) = \begin{cases} \log(x + 0.5) & x \ge 0 \\ -\mathrm{softplus}(-x) & x < 0 \end{cases}$$

Both branches meet at $0.5$, so $g$ is continuous at zero. Two consequences for
lookup: the log-space forward pass computes a **different cell** from the
default linear-candidate one, so an equivalence check must pass `candidate=g`;
and it requires $a^{\langle 0\rangle} \ge 0$, rejecting a negative initial state
rather than returning `nan`. $\mathrm{softplus}$ is evaluated as
$\log(1 + e^{-\lvert z\rvert}) + \max(z, 0)$, which never exponentiates a
positive number.

## Framework divergences

The same layer name denotes two different functions across the ecosystem. The
divergence is the **reset gate's placement**. This hub, Cho et al. (2014) and Ng
(2018) apply it before the hidden transform; PyTorch computes

$$n_t = \tanh\!\big(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{t-1} + b_{hn})\big),$$

applying the reset **after** the recurrent transform, which is the
cuDNN-compatible form Keras exposes as `reset_after=True`. The two are not
algebraically equal, and `torch.nn.GRU` offers no flag to change it.

Measured on 2026-09-19 against `tf-nightly 2.22.0.dev20260912` /
`keras-nightly 3.16.0.dev` and `torch 2.14.0+cpu`, in this hub's
$(n_x, m, T_x)$ convention with `return_sequences=True`:

| Framework cell | Matches the equations above? | Measured max abs error |
| --- | --- | --- |
| `tf.keras.layers.LSTM` | yes | `2.2e-16` |
| `tf.keras.layers.GRU(reset_after=False)` | yes | `2.2e-16` |
| `tf.keras.layers.GRU(reset_after=True)` | **no** — cuDNN reset-after form | `1.79` |
| `torch.nn.GRU` | **no** — reset-after form, no flag to change it | — |

`torch.nn.LSTM` computes the gate equations above with the bias split into
`b_ih` and `b_hh`, whose sum is the single $b_g$ here; only the TensorFlow side
has been measured against this reference.

## TensorFlow parity mapping

The contract the TensorFlow parity port is held to. That port is
[`dlhub/tensorflow/sequence/lstm_gru.py`](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/tensorflow/sequence/lstm_gru.py);
the mapping is recorded here because it is lookup material either way.

| Item | Value |
| --- | --- |
| Layer construction | `reset_after=False`, `activation="tanh"`, `recurrent_activation="sigmoid"`, `return_sequences=True` |
| Matrix orientation | Keras computes `inputs @ kernel`, so its matrices are the **transpose** of this hub's: split $W_g$ at column $n_a$, then `kernel` $\leftarrow W_{gx}^{\top}$ and `recurrent_kernel` $\leftarrow W_{ga}^{\top}$ |
| Packed gate order | GRU `[z, r, h]`; LSTM `[i, f, c, o]`, where Keras' `i` is this hub's $\Gamma_u$ |
| GRU shapes | `kernel` $(n_x,\, 3 n_a)$, `recurrent_kernel` $(n_a,\, 3 n_a)$, `bias` $(3 n_a,)$ — a single bias row, because `reset_after=False` |
| LSTM shapes | the same, with $4 n_a$ throughout |
| GRU sign flip | Keras writes the blend as $h = z \odot h_{t-1} + (1 - z) \odot \tilde{h}$, putting $z$ on the **old** state, so $z = 1 - \Gamma_u$. Both are $\sigma(\cdot)$, so the pre-activation is negated: $W_z \leftarrow -W_u$, $b_z \leftarrow -b_u$, and the gradient carries the same flip, $dW_u = -dW_z$ |
| LSTM sign flips | none |
| Input and state bridging | Keras is batch-first $(m, T_x, n_x)$, so transpose $x$ by `(1, 2, 0)`; initial states are $a_0^{\top}$ and $c_0^{\top}$, each $(m, n_a)$ |
| dtype | `float64`, set per layer with `dtype="float64"` at construction (the hub port's choice) or process-wide with `tf.keras.backend.set_floatx("float64")` before the layer is built; float32 cannot carry the gradient tolerance |

## Fixture and tolerances

Every implementation and test shares one deterministic fixture per module, built
by that module's `make_fixture()`. Dimensions are reused from the vanilla RNN
fixture so the two cell families can be contrasted on identical shapes.

- Seed: `np.random.seed(1)`
- dtype: `float64` (float32 is too coarse for a finite-difference gradient check)
- Dimensions: $n_x = 3$, $n_a = 5$, $n_y = 2$, $m = 10$, $T_x = 4$ — except the
  minimal cells, which use $n_x = 3$, $n_a = 5$, $m = 10$, and **$T_x = 7$**,
  deliberately odd and not a power of two

The draw order is recorded so the arrays reproduce bit for bit, and the one-hot
targets are drawn **last** so adding them cannot disturb the arrays a
forward-parity check compares.

| Module | Draw order |
| --- | --- |
| LSTM | `x`, `a0`, `c0`, `Wf`, `Wu`, `Wc`, `Wo`, `Wy`, `bf`, `bu`, `bc`, `bo`, `by`, then the one-hot `y` |
| GRU | `x`, `a0`, `Wr`, `Wu`, `Wc`, `Wy`, `br`, `bu`, `bc`, `by`, then the one-hot `y` |
| Minimal cells | `x`, `a0`, then minGRU's `Wu`, `Wc` and minLSTM's `Wf`, `Wu`, `Wc`, each $(n_a, n_x)$ |

| Check | Compared quantities | Tolerance |
| --- | --- | --- |
| Forward parity | activations, cell states, and outputs | `atol = 1e-8` |
| BPTT gradient check | analytic gradients vs. central finite differences, for every parameter plus $da_0$, $dc_0$ and $dx$ | relative error $< 10^{-7}$ |
| TensorFlow parity | `GradientTape` gradients vs. the NumPy BPTT, from the identical fixture arrays in float64 | relative error $< 10^{-6}$ |
| Scan equivalence | parallel or log-space scan vs. the sequential recurrence | `atol = 1e-10` |
| Discrimination | a flipped sign, which the metric must reject | each suite keeps one |

The relative error is norm-based,
$\lVert a - n \rVert / (\lVert a \rVert + \lVert n \rVert)$, which is bounded by
$1$, so a flipped sign scores near $1$ rather than infinity.

**The finite-difference step is not the shared default.** The helper's
`FD_EPSILON` is $10^{-7}$ and the vanilla RNN suite uses it, but both gated
suites pass $\varepsilon = 10^{-6}$. The reason is a floor, not a fudge:
subtracting two nearly equal float64 losses puts a floor of roughly
$\mathrm{ulp}(L) / (2 \varepsilon \lVert g \rVert)$ on the relative error of a
gradient $g$, and the GRU's reset-gate gradient has norm $0.06$ on this fixture,
which at $\varepsilon = 10^{-7}$ floors it near $1.2 \times 10^{-7}$ — above the
declared tolerance however correct the analytic gradient is. At $10^{-6}$ the
floor drops tenfold while the truncation error it trades against stays near
$10^{-12}$, and every gradient lands at or under $1.1 \times 10^{-8}$. Central
differences are required either way: the one-sided form's $O(\varepsilon)$ error
would fail a correct implementation at this tolerance.

The finite-difference machinery — `numeric_gradient`, `relative_error`,
`global_norm`, and the shared `FD_EPSILON` — lives in
`dlhub.nn.sequence._gradient_check`, shared by the three sequence suites so they cannot disagree about what "agrees" means. It is a
deliberately private, per-tensor test fixture, not the hub's taught,
whole-network gradient check (`dlhub.training.gradient_checking`, which has its
own pages); the two carry different APIs on purpose, and the per-tensor one is
not part of the public API.

## Failure modes

Explicit rejections, all `ValueError`:

| Condition | Raised by |
| --- | --- |
| `x` is not three-dimensional | either forward pass, and the minimal cells' per-timestep projection |
| `a0` does not have shape $(n_a, m)$ | either forward pass, and all three scans |
| `c0` does not have shape $(n_a, m)$ | the LSTM forward pass |
| $\alpha$ and $\beta$ (or their logs) disagree in shape | the plain-space and log-space scans |
| any entry of `a0` is negative | the log-space scan, which cannot take its logarithm |
| a gate matrix has no more than $n_a$ columns, or is not 2-D | the column split, which would otherwise return an empty input half |

Silent failure modes, which raise nothing and must be tested for:

| Symptom | Cause |
| --- | --- |
| Framework output differs by order $0.1$ or more rather than by rounding | the GRU reset placement; `reset_after=True` measures `1.79` max abs error against this reference |
| Forward pass matches a framework but gradients do not | a sign flip applied to the weight and not to the gradient; $dW_u = -dW_z$ |
| A gradient check that passes in float32 | it cannot legitimately; the check requires float64 |
| A parallel scan that passes at $T_x = 4$ and fails at $T_x = 7$ | a hidden power-of-two length assumption |
| Minimal-cell state underflowing to zero over long sequences | the prefix product of $\alpha \in (0, 1)$; use the log-space scan, with $a^{\langle 0\rangle} \ge 0$ |
| A loss that diverges or goes `NaN` while gates look healthy | exploding gradients, which gating does not address; clip by global norm |

## Implementation Examples

### Complete Implementations:

> #### **[LSTM — from scratch (NumPy)](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/nn/sequence/lstm.py)** - The three gates, the separate cell state, forward propagation through time, and the hand-derived BPTT, computed term by term and gradient-checked against central finite differences on the shared fixture.
> #### **[GRU — from scratch (NumPy)](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/nn/sequence/gru.py)** - The two-gate cell in the Cho/Ng form, with the reset gate applied before the hidden transform, and the same gradient-checked BPTT. Its module notes record the reset-placement divergence tabulated above.
> #### **[minGRU and minLSTM — from scratch (NumPy)](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/nn/sequence/min_gated.py)** - The 2024 minimal variants in all three forms — sequential recurrence, plain-space associative scan, and the stable log-space scan — checked against each other and against the classical cells they reduce from. Derives no BPTT by design.
> #### **[LSTM and GRU — TensorFlow parity port](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/tensorflow/sequence/lstm_gru.py)** - Both cells as `tf.keras.layers.LSTM` and `tf.keras.layers.GRU` with a `Dense` + softmax read-out, differentiated by `tf.GradientTape`. Reproduces the NumPy references on the shared fixtures — activations, outputs, and the final cell state to `atol=1e-8`, and every gradient to a relative error below `1e-15` against the analytic BPTT — and tabulates the Keras conventions that must be reconciled first: `reset_after=False`, the inverted update gate `z = 1 - Γ_u`, the packed gate orders `[z, r, h]` and `[i, f, c, o]`, and the `unit_forget_bias` default.

Rendered signatures and docstrings for the three NumPy modules are in
the generated
[neural-network API reference](api/nn.md), which is built from the package
rather than copied here.

## References

- **Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. _Neural Computation_ 9(8), 1735–1780.** – The original LSTM, and the source of the input gate this page writes $\Gamma_u$.
- **Gers, F. A., Schmidhuber, J. & Cummins, F. (2000). Learning to Forget: Continual Prediction with LSTM. _Neural Computation_ 12(10), 2451–2471.** – The forget gate, which the 1997 cell lacked and every modern LSTM has.
- **Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H. & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. EMNLP.**  
  [https://arxiv.org/abs/1406.1078](https://arxiv.org/abs/1406.1078) – The GRU, in the reset-before-transform form these equations follow.
- **Chung, J., Gulcehre, C., Cho, K. & Bengio, Y. (2014). Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling.**  
  [https://arxiv.org/abs/1412.3555](https://arxiv.org/abs/1412.3555) – The GRU-versus-LSTM comparison, which reached no conclusive verdict between them.
- **Feng, L., Tung, F., Ahmed, M. O., Bengio, Y. & Hajimirsadeghi, H. (2024). Were RNNs All We Needed?**  
  [https://arxiv.org/abs/2410.01201](https://arxiv.org/abs/2410.01201) – minGRU and minLSTM, the parallel scan, and the log-space implementation of Sec. B.1 with the positive candidate of Sec. B.2.1.
- **Bengio, Y., Simard, P. & Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult. _IEEE Transactions on Neural Networks_ 5(2), 157–166.** – The vanishing-gradient analysis the carry path answers.
- **Pascanu, R., Mikolov, T. & Bengio, Y. (2013). On the difficulty of training Recurrent Neural Networks. ICML.**  
  [https://arxiv.org/abs/1211.5063](https://arxiv.org/abs/1211.5063) – Exploding gradients and global-norm clipping, which gating does not replace.
- **Goodfellow, I., Bengio, Y. & Courville, A. (2016). _Deep Learning_, Ch. 10.**  
  [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/) – Reference treatment of gated recurrent architectures.
- **Ng, A. (2018). Deep Learning Specialization, Course 5 (Sequence Models), Week 1.**  
  [https://www.coursera.org/specializations/deep-learning](https://www.coursera.org/specializations/deep-learning) – The notation used here, including $\Gamma_u$ for both cells' update gate.
- **Keras `GRU` layer documentation.**  
  [https://keras.io/api/layers/recurrent_layers/gru/](https://keras.io/api/layers/recurrent_layers/gru/) – The `reset_after` flag, its cuDNN-compatible default, and the packed gate order.
- **Keras `LSTM` layer documentation.**  
  [https://keras.io/api/layers/recurrent_layers/lstm/](https://keras.io/api/layers/recurrent_layers/lstm/) – The LSTM layer's weights and gate packing.
- **PyTorch `torch.nn.GRU` documentation.**  
  [https://pytorch.org/docs/stable/generated/torch.nn.GRU.html](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html) – The reset-after recurrence, with no option to change it.

## Crosswalk

| Reader wants to | Go to |
| --- | --- |
| Learn it step by step | [LSTM and GRU (notebook)](../tutorials/02-lstm-and-gru.ipynb) |
| Do it in a project | [Choose a gated cell](../how-to/choose-a-gated-cell.md) |
| Look up an equation or shape | LSTM and GRU — *this page* |
| Understand why it works | [LSTM and GRU](../explanation/lstm-and-gru.md) |
| Learn by running it | [LSTM and GRU (notebook)](../tutorials/02-lstm-and-gru.ipynb) |

## Key Takeaways

1. The GRU runs two gates over one state: $\Gamma_r$ decides how much of
   $a^{\langle t-1\rangle}$ the candidate sees, $\Gamma_u$ blends candidate and
   previous state convexly, and $a^{\langle t\rangle} = c^{\langle t\rangle}$.
   The LSTM runs three gates over two states, adding the cell state
   $c^{\langle t\rangle} = \Gamma_u \odot \tilde{c} + \Gamma_f \odot c^{\langle t-1\rangle}$
   and exposing it as $a^{\langle t\rangle} = \Gamma_o \odot \tanh(c^{\langle t\rangle})$.
2. Both cells use the concatenated weight form $W_g = [\,W_{ga} \mid W_{gx}\,]$,
   shape $(n_a,\, n_a + n_x)$, acting on
   $[\,a^{\langle t-1\rangle};\,x^{\langle t\rangle}\,]$; splitting at column
   $n_a$ recovers the recurrent and input halves. The gate order is forget,
   update, candidate, output.
3. Every gate derivative is read off the cached gate,
   $\sigma'(z) = \Gamma \odot (1 - \Gamma)$, and every shared weight accumulates
   its gradient over all $T_x$ steps. The LSTM threads two recurrent gradients
   backward, the GRU one; the GRU deposits gradient into
   $a^{\langle t-1\rangle}$ along three paths that are summed.
4. The one derivative that defines the family is
   $\partial c^{\langle t\rangle} / \partial c^{\langle t-1\rangle} = \Gamma_f^{\langle t\rangle}$
   — elementwise, with no weight matrix and no saturating nonlinearity in the way
   — against the vanilla RNN's repeated Jacobian product bounded by
   $\lVert W_{aa}\rVert_2^{\,k}$. The GRU's $(1 - \Gamma_u)$ blend path is the
   same mechanism.
5. A cell costs its gate count times $n_a (n_a + n_x + 1)$, so 135 parameters
   for a GRU and 180 for an LSTM at $n_x = 3$, $n_a = 5$: quadratic in $n_a$,
   linear in $n_x$.
6. minGRU and minLSTM delete the gates' dependence on $a^{\langle t-1\rangle}$,
   which turns the cell into $a^{\langle t\rangle} = \alpha \odot a^{\langle t-1\rangle} + \beta$
   and lets an associative scan resolve it in $\lceil \log_2 T_x \rceil$ rounds.
   They also drop the reset gate, the output gate, the biases, the recurrent
   weight half, and the candidate's $\tanh$ — every one deliberately.
7. The log-space scan is a different cell, not just a different arithmetic: its
   candidate is $g$ rather than the identity, and it requires
   $a^{\langle 0\rangle} \ge 0$. Compare it against the sequential pass with
   `candidate=g`, at `atol = 1e-10`.
8. "GRU" names two functions. These equations apply the reset gate **before**
   the hidden transform; `torch.nn.GRU` and
   `tf.keras.layers.GRU(reset_after=True)` apply it after, measured at `1.79`
   max abs error against `2.2e-16` for `reset_after=False`.
9. Parity is fixed by deterministic `float64` fixtures (seed `1`;
   $n_x, n_a, n_y, m, T_x = 3, 5, 2, 10, 4$, and $T_x = 7$ for the minimal
   cells), with forward matches to `atol = 1e-8` and analytic-versus-numeric
   gradients to relative error below $10^{-7}$ — taken at
   $\varepsilon = 10^{-6}$, because $10^{-7}$ floors the smallest gradient above
   the tolerance.
