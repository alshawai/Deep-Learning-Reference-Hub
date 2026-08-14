# Activation Functions and Their Derivatives

These derivatives are used during [Forward and Backward Propagation](../explanation/forward-and-backward-propagation.md).

## ReLU Activation

$$g(z) = \max(0, z)$$
$$g'(z) = \begin{cases}
1 & \text{if } z > 0 \\
0 & \text{if } z \leq 0
\end{cases}$$

## Sigmoid Activation

$$g(z) = \sigma(z) = \frac{1}{1 + e^{-z}}$$
$$g'(z) = \sigma(z)(1 - \sigma(z))$$

## Hyperbolic Tangent

$$g(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$
$$g'(z) = 1 - \tanh^2(z)$$

## Leaky ReLU

$$g(z) = \begin{cases}
z & \text{if } z > 0 \\
\alpha z & \text{if } z \leq 0
\end{cases}$$
$$g'(z) = \begin{cases}
1 & \text{if } z > 0 \\
\alpha & \text{if } z \leq 0
\end{cases}$$
