# Recurrent neural networks

This page is the lookup reference for the vanilla (simple) recurrent network: the
recurrence relation, forward propagation through time, backpropagation through
time (BPTT), the architecture types by input/output cardinality, the
vanishing- and exploding-gradient failure mode with its clipping remedy, and the
bidirectional and deep compositions. Notation follows Andrew Ng's Deep Learning
Specialization, Course 5, Week 1, verbatim, so the sibling sequence references
extend the same symbols. Gated cells (LSTM, GRU) are out of scope here; this page
states the vanishing-gradient problem that motivates them and stops there.

For the reasoning behind the mechanics catalogued here, see the
[explanation of recurrent neural networks](../explanation/recurrent-neural-networks.md).

## Contents

- [Notation and shapes](#notation-and-shapes)
- [Forward recurrence](#forward-recurrence)
- [Forward propagation through time](#forward-propagation-through-time)
- [Loss](#loss)
- [Backpropagation through time](#backpropagation-through-time)
- [Architecture types](#architecture-types)
- [Vanishing and exploding gradients](#vanishing-and-exploding-gradients)
- [Bidirectional RNNs](#bidirectional-rnns)
- [Deep stacked RNNs](#deep-stacked-rnns)
- [PyTorch parity](#pytorch-parity)
- [Fixture and tolerances](#fixture-and-tolerances)
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
| $m$ | batch size (number of examples) |
| $T_x$ | number of input timesteps |
| $T_y$ | number of output timesteps ($T_y = T_x$ in the many-to-many equal-length case this module implements) |

Every array is stored with the feature axis first and time last, so a single
timestep is a plain 2-D slice.

| Symbol | Shape |
| --- | --- |
| $x$ | $(n_x, m, T_x)$ |
| $a$, hidden states | $(n_a, m, T_x)$ |
| $\hat{y}$ | $(n_y, m, T_x)$ |
| $W_{ax}$ | $(n_a, n_x)$ |
| $W_{aa}$ | $(n_a, n_a)$ |
| $W_{ya}$ | $(n_y, n_a)$ |
| $b_a$ | $(n_a, 1)$ |
| $b_y$ | $(n_y, 1)$ |

The weights $W_{aa}$, $W_{ax}$, $W_{ya}$, $b_a$, $b_y$ are **shared across every
timestep**; there is one copy, reused at each step.

## Forward recurrence

For $t = 1, \ldots, T_x$, the cell updates the hidden state and reads out a
distribution over classes:

$$a^{\langle t\rangle} = \tanh\!\big(W_{aa}\,a^{\langle t-1\rangle} + W_{ax}\,x^{\langle t\rangle} + b_a\big)$$

$$\hat{y}^{\langle t\rangle} = \mathrm{softmax}\!\big(W_{ya}\,a^{\langle t\rangle} + b_y\big)$$

The hidden activation is $\tanh$; the output is a column-wise $\mathrm{softmax}$
over the $n_y$ classes. The equivalent stacked form concatenates the recurrent
and input weights into one matrix acting on the stacked state and input:

$$a^{\langle t\rangle} = \tanh\!\big(W_{a}\,[\,a^{\langle t-1\rangle};\,x^{\langle t\rangle}\,] + b_a\big), \qquad W_{a} = [\,W_{aa}\mid W_{ax}\,]$$

The two forms are numerically identical. The implementation keeps $W_{aa}$ and
$W_{ax}$ separate so the two weight-gradient terms stay distinct.

## Forward propagation through time

Unrolling the cell threads the hidden state from one step to the next, starting
from $a^{\langle 0\rangle} = \mathbf{0}$ (the zero vector) by convention. The full
pass consumes $x$ of shape $(n_x, m, T_x)$ and produces the stacked hidden states
$a$ of shape $(n_a, m, T_x)$ and outputs $\hat{y}$ of shape $(n_y, m, T_x)$.

## Loss

The total loss is summed over timesteps,

$$L = \sum_{t=1}^{T_x} L^{\langle t\rangle},$$

with each per-step term the batch-mean softmax cross-entropy of the prediction
against the one-hot target,

$$L^{\langle t\rangle} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{c=1}^{n_y} y^{\langle t\rangle}_{c,i} \, \log \hat{y}^{\langle t\rangle}_{c,i}.$$

## Backpropagation through time

BPTT walks the unrolled graph from $t = T_x$ back to $t = 1$. The gradient
arriving at $a^{\langle t\rangle}$ has **two sources**: the read-out at step $t$
and the recurrence carried back from step $t+1$. Per timestep, with $\odot$ the
elementwise product and sums taken over the batch axis:

$$dz_y^{\langle t\rangle} = \frac{1}{m}\big(\hat{y}^{\langle t\rangle} - y^{\langle t\rangle}\big)$$

$$dW_{ya}^{\langle t\rangle} = dz_y^{\langle t\rangle}\,\big(a^{\langle t\rangle}\big)^{\!\top}, \qquad db_y^{\langle t\rangle} = \sum_{\text{batch}} dz_y^{\langle t\rangle}$$

$$da^{\langle t\rangle} = W_{ya}^{\!\top}\, dz_y^{\langle t\rangle} \;+\; \big(da^{\langle t\rangle}\big)_{\text{from } t+1}$$

The $\tanh$ backward reads its derivative straight off the cached activation,
$\tfrac{d}{dz}\tanh(z) = 1 - \tanh^2(z) = 1 - (a^{\langle t\rangle})^2$:

$$dz_a^{\langle t\rangle} = \big(1 - (a^{\langle t\rangle})^2\big) \odot da^{\langle t\rangle}$$

$$db_a^{\langle t\rangle} = \sum_{\text{batch}} dz_a^{\langle t\rangle}, \quad dW_{ax}^{\langle t\rangle} = dz_a^{\langle t\rangle}\,\big(x^{\langle t\rangle}\big)^{\!\top}, \quad dW_{aa}^{\langle t\rangle} = dz_a^{\langle t\rangle}\,\big(a^{\langle t-1\rangle}\big)^{\!\top}$$

$$\big(da^{\langle t-1\rangle}\big)_{\text{from } t} = W_{aa}^{\!\top}\, dz_a^{\langle t\rangle}$$

Because the weights are shared, each parameter gradient is the **sum of its
per-step contributions**,

$$dW_{ax} = \sum_{t=1}^{T_x} dW_{ax}^{\langle t\rangle}, \quad dW_{aa} = \sum_{t=1}^{T_x} dW_{aa}^{\langle t\rangle}, \quad dW_{ya} = \sum_{t=1}^{T_x} dW_{ya}^{\langle t\rangle},$$

and likewise for $db_a$ and $db_y$. After the loop the recurrence gradient has
reached the initial state, giving $da^{\langle 0\rangle}$; the gradient with
respect to the input is $dx$ of shape $(n_x, m, T_x)$, whose step $t$ slice is
$W_{ax}^{\!\top}\, dz_a^{\langle t\rangle}$.

## Architecture types

The cardinality of a task fixes how many steps take an input and how many emit a
read-out. Every type below reuses the **same cell**; none needs new mathematics
(Ng, 2018).

| Architecture | Input/output | Typical task |
| --- | --- | --- |
| one-to-one | $T_x = T_y = 1$ | ordinary feed-forward classification (degenerate RNN) |
| one-to-many | $T_x = 1$, $T_y > 1$ | sequence generation from a single input |
| many-to-one | $T_x > 1$, $T_y = 1$ | sequence classification, e.g. sentiment |
| many-to-many, equal length | $T_x = T_y$ | per-step labelling, e.g. named-entity recognition — the case this module implements |
| many-to-many, unequal length | $T_x \ne T_y$ | encoder-decoder transduction, e.g. machine translation |

## Vanishing and exploding gradients

The one-step Jacobian of the recurrence is

$$\frac{\partial a^{\langle t\rangle}}{\partial a^{\langle t-1\rangle}} = \operatorname{diag}\!\big(1 - (a^{\langle t\rangle})^2\big)\,W_{aa}.$$

Propagating a gradient back $k$ steps multiplies $k$ of these Jacobians:

$$\frac{\partial a^{\langle t\rangle}}{\partial a^{\langle t-k\rangle}} = \prod_{i=t-k+1}^{t} \operatorname{diag}\!\big(1 - (a^{\langle i\rangle})^2\big)\,W_{aa}.$$

Since $0 < 1 - (a)^2 \le 1$ for $\tanh$, each diagonal factor has spectral norm at
most $1$, so the product's norm is bounded by $\lVert W_{aa}\rVert_2^{\,k}$. When
the largest singular value of $W_{aa}$ is below $1$, the contribution decays
geometrically in $k$ — **vanishing gradients**, which make long-range
dependencies hard to learn (Bengio, Simard & Frasconi, 1994). When it exceeds
$1$, the bound permits geometric growth — **exploding gradients**, which
destabilize training (Pascanu, Mikolov & Bengio, 2013).

Exploding gradients are controlled by **global-norm clipping**: compute one L2
norm over all parameter gradients stacked into a single vector, and, if it
exceeds a threshold $v$, rescale every gradient by the same factor. This caps the
magnitude while preserving direction (Pascanu, Mikolov & Bengio, 2013):

$$\lVert g\rVert_2 = \sqrt{\textstyle\sum_{\theta} \lVert d\theta\rVert_2^2}, \qquad g \leftarrow \begin{cases} \dfrac{v}{\lVert g\rVert_2}\, g & \text{if } \lVert g\rVert_2 > v, \\[4pt] g & \text{otherwise.} \end{cases}$$

```python
# Global-norm clipping: one norm over every parameter gradient at once.
total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients.values()))
if total_norm > max_norm:
    scale = max_norm / total_norm
    gradients = {name: g * scale for name, g in gradients.items()}
```

Clipping addresses only the exploding case. Vanishing gradients are the standing
motivation for the gated cells (LSTM, GRU) documented in the sibling
`lstm-and-gru` topic; as of 2025 gated cells and attention-based models
(Vaswani et al., 2017) are the usual answer to long-range dependence.

## Bidirectional RNNs

A bidirectional RNN runs **two independent recurrences** over the sequence — a
forward pass $\overrightarrow{a}^{\langle t\rangle}$ (left to right) and a
backward pass $\overleftarrow{a}^{\langle t\rangle}$ (right to left) — each a full
application of the same cell. The representation at step $t$ is the
concatenation

$$\big[\,\overrightarrow{a}^{\langle t\rangle};\ \overleftarrow{a}^{\langle t\rangle}\,\big],$$

of width $2 n_a$, so the stacked hidden states have shape $(2 n_a, m, T_x)$. A
read-out layer then maps the $2 n_a$-dimensional concatenation to $n_y$. There is
no new cell mathematics (Schuster & Paliwal, 1997). A bidirectional model needs
the whole sequence before it can emit any output.

## Deep stacked RNNs

A deep RNN stacks $L$ recurrent layers. Layer $l$ is itself a full recurrence over
the sequence produced by the layer below, consuming the activation
$a^{[l-1]\langle t\rangle}$ in place of the input $x^{\langle t\rangle}$ (layer $1$
consumes $x$). Only the top layer's read-out is a model output. Layer $l$'s input
weight has shape $(n_a^{[l]}, n_a^{[l-1]})$, with $n_a^{[0]} = n_x$. This is again
a composition of the same cell, with no new cell mathematics.

## PyTorch parity

The framework parity port targets `torch.nn.RNN`, whose recurrence (as of PyTorch
2.x) uses **two** bias vectors and emits only hidden states — the class read-out
is a separate `nn.Linear`:

$$h_t = \tanh\!\big(W_{ih}\,x_t + b_{ih} + W_{hh}\,h_{t-1} + b_{hh}\big).$$

To reproduce this module's single-bias recurrence and softmax read-out exactly,
map the parameters as follows:

| This module | `torch.nn.RNN` / `nn.Linear` |
| --- | --- |
| $W_{ax}$ | `weight_ih` |
| $W_{aa}$ | `weight_hh` |
| $b_a$ | `bias_ih` |
| — | `bias_hh` $= \mathbf{0}$ |
| $W_{ya}$ | `Linear.weight` |
| $b_y$ | `Linear.bias` |

Parity is checked on **gradients**, not only on the forward pass.

## Fixture and tolerances

Every implementation and test shares one deterministic fixture, built by
`make_fixture()`. Parity is checked against it, never against whole training
curves.

- Seed: `np.random.seed(1)`
- dtype: `float64` (float32 is too coarse for a finite-difference gradient check)
- Dimensions: $n_x = 3$, $n_a = 5$, $n_y = 2$, $m = 10$, $T_x = 4$

The draw order is recorded so the arrays reproduce bit for bit; the one-hot
targets are drawn **after** the seven parameter arrays so adding them cannot
disturb the arrays a forward-parity check compares.

| # | Array | Shape |
| --- | --- | --- |
| 1 | `x` | $(n_x, m, T_x)$ |
| 2 | `a0` | $(n_a, m)$ |
| 3 | `Wax` | $(n_a, n_x)$ |
| 4 | `Waa` | $(n_a, n_a)$ |
| 5 | `Wya` | $(n_y, n_a)$ |
| 6 | `ba` | $(n_a, 1)$ |
| 7 | `by` | $(n_y, 1)$ |

| Check | Compared quantities | Tolerance |
| --- | --- | --- |
| Forward parity | activations $a^{\langle t\rangle}$ and outputs $\hat{y}^{\langle t\rangle}$ | `atol = 1e-8` |
| BPTT parity | analytic gradients vs. central finite differences ($\varepsilon = 10^{-7}$) | relative error $< 10^{-7}$ |
| PyTorch parity | autograd gradients vs. NumPy BPTT | relative error $< 10^{-6}$ |

## Implementation Examples

### Complete Implementations:
> #### **[Vanilla RNN — from scratch (NumPy)](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/nn/sequence/rnn.py)** - Forward propagation through time, term-by-term BPTT, global-norm gradient clipping, and the bidirectional and deep compositions, all checked against the shared fixture. Rendered signatures and docstrings are in the generated [neural-network API reference](api/nn.md).
> #### **[Vanilla RNN — PyTorch parity port](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/src/dlhub/pytorch/sequence/rnn.py)** - The same recurrence built on `torch.nn.RNN` with an `nn.Linear` + softmax read-out, showing the weight mapping and the two-bias convention `nn.RNN` uses in place of Ng's single `b_a`. Checked against the NumPy reference on the shared fixture: forward activations and outputs to `atol=1e-8`, and autograd gradients versus the analytic BPTT to relative error below `1e-6`.

## References

- **Rumelhart, D. E., Hinton, G. E. & Williams, R. J. (1986). Learning representations by back-propagating errors. _Nature_ 323, 533–536.** – The backpropagation algorithm that BPTT unrolls through time.
- **Elman, J. L. (1990). Finding Structure in Time. _Cognitive Science_ 14(2), 179–211.** – The simple recurrent network this page documents.
- **Bengio, Y., Simard, P. & Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult. _IEEE Transactions on Neural Networks_ 5(2), 157–166.** – The original vanishing-gradient analysis.
- **Schuster, M. & Paliwal, K. K. (1997). Bidirectional Recurrent Neural Networks. _IEEE Transactions on Signal Processing_ 45(11), 2673–2681.** – The bidirectional construction.
- **Pascanu, R., Mikolov, T. & Bengio, Y. (2013). On the difficulty of training Recurrent Neural Networks. ICML.**  
  [https://arxiv.org/abs/1211.5063](https://arxiv.org/abs/1211.5063) – Exploding gradients and gradient clipping.
- **Goodfellow, I., Bengio, Y. & Courville, A. (2016). _Deep Learning_, Ch. 10.**  
  [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/) – Reference derivation of forward propagation and BPTT.
- **Ng, A. (2018). Deep Learning Specialization, Course 5 (Sequence Models), Week 1.**  
  [https://www.coursera.org/specializations/deep-learning](https://www.coursera.org/specializations/deep-learning) – Source of the notation used here.
- **Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.**  
  [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) – Where sequence modeling leads beyond the RNN.

## Crosswalk

Live links are added as each sibling ships; entries marked *planned* are not yet
published.

| Reader wants to | Go to |
| --- | --- |
| Learn it step by step | [Recurrent neural networks (notebook)](../tutorials/01-recurrent-neural-networks.ipynb) |
| Do it in a project | Not written for this topic; see the language-modeling-and-sampling topic |
| Look up an equation or shape | Recurrent neural networks — *this page* |
| Understand why it works | [Recurrent neural networks](../explanation/recurrent-neural-networks.md) |
| Learn by running it | [Recurrent neural networks (notebook)](../tutorials/01-recurrent-neural-networks.ipynb) |

## Key Takeaways

1. A vanilla RNN applies one shared cell at every timestep: $a^{\langle t\rangle} = \tanh(W_{aa} a^{\langle t-1\rangle} + W_{ax} x^{\langle t\rangle} + b_a)$, with the softmax read-out $\hat{y}^{\langle t\rangle} = \mathrm{softmax}(W_{ya} a^{\langle t\rangle} + b_y)$ and $a^{\langle 0\rangle} = \mathbf{0}$.
2. Arrays carry the feature axis first and time last, so $x$ is $(n_x, m, T_x)$, hidden states are $(n_a, m, T_x)$, and outputs are $(n_y, m, T_x)$.
3. BPTT accumulates each shared weight's gradient over all timesteps, and the gradient at $a^{\langle t\rangle}$ merges two sources: the read-out at $t$ and the recurrence from $t+1$.
4. The four cardinalities — one-to-one, one-to-many, many-to-one, and many-to-many (equal or unequal length) — all reuse the same cell.
5. Backpropagating $k$ steps multiplies $k$ Jacobians bounded by $\lVert W_{aa}\rVert_2^{\,k}$: a largest singular value below $1$ vanishes the gradient, above $1$ explodes it. Global-norm clipping caps the exploding case; vanishing motivates the gated cells.
6. Bidirectional and deep RNNs are compositions of the same cell — a bidirectional layer concatenates two direction passes to width $2 n_a$, and a deep layer feeds one layer's activations to the next.
7. Parity is fixed by one deterministic `float64` fixture (seed `1`; $n_x, n_a, n_y, m, T_x = 3, 5, 2, 10, 4$), with forward matches to `atol = 1e-8` and analytic-vs-numeric gradients to relative error $< 10^{-7}$.
