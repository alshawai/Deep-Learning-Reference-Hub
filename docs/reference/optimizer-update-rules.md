# Optimizer Update Rules

This page states the gradient-descent variants, update equations, defaults, and implementation references. See [optimization algorithms](../explanation/optimization-algorithms.md) for the reasoning behind the methods and [how to choose an optimizer](../how-to/choose-an-optimizer.md) for recommendations.

## Batch Gradient Descent

Batch gradient descent uses the entire training dataset for each update.

```python
for epoch in range(num_epochs):
    gradient = compute_gradient(X_train, y_train, theta)
    theta -= learning_rate * gradient
```

$$\theta_{t+1} = \theta_t - \alpha \frac{1}{m} \sum_{i=1}^{m} \nabla_\theta J(\theta_t, x^{(i)}, y^{(i)})$$

## Stochastic Gradient Descent

Stochastic gradient descent (SGD) uses one training example at a time.

```python
for epoch in range(num_epochs):
    X_shuffled, y_shuffled = shuffle(
        X_train, y_train
    )  # To ensure randomness in data pattern

    for i in range(m):  # No. of training examples
        gradient = compute_gradient(X_shuffled[i], y_shuffled[i], theta)
        theta -= learning_rate * gradient
```

$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta J(\theta_t, x^{(i)}, y^{(i)})$$

## Mini-batch Gradient Descent

Mini-batch gradient descent uses small batches of training examples.

```python
for epoch in range(num_epochs):
    mini_batches = create_mini_batches(
        X_train, y_train, batch_size
    )  # To ensure random batches

    for mini_batch in mini_batches:
        X_batch, y_batch = mini_batch
        gradient = compute_gradient(X_batch, y_batch, theta)
        theta -= learning_rate * gradient
```

$$\theta_{t+1} = \theta_t - \alpha \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \nabla_\theta J(\theta_t, x^{(i)}, y^{(i)})$$

Here $\mathcal{B}$ is the mini-batch and $|\mathcal{B}|$ is the batch size.

Suggested batch sizes:

- **Small datasets**: 32-64 samples.
- **Medium datasets**: 128-256 samples.
- **Large datasets**: 256-512 samples.
- **Very large datasets**: 512-1024 samples.

Powers of 2 work well with GPU memory architecture. The implementation is available in [`mini_batch.py`](https://github.com/eima40x4c/Deep-Learning-Reference-Hub/blob/main/src/dlhub/optimizers/mini_batch.py).

## Momentum

Momentum updates parameters using a velocity $v_t$ rather than the current gradient alone.

```python
v = np.zeros_like(theta)  # Initialize velocity
for t in range(num_iterations):
    gradient = compute_gradient(X_batch, y_batch, theta)

    v = beta * v + (1 - beta) * gradient
    theta -= learning_rate * v
```

$$\begin{align}
v_t &= \beta v_{t-1} + (1-\beta) \nabla_\theta J(\theta_t) \\
\theta_{t+1} &= \theta_t - \alpha v_t
\end{align}$$

Typical choices are $\beta = 0.9$ for a standard setting, $\beta = 0.99$ for noisier gradients, and $\beta = 0.5$ for rapidly changing landscapes. See [`momentum.py`](https://github.com/eima40x4c/Deep-Learning-Reference-Hub/blob/main/src/dlhub/optimizers/momentum.py).

## RMSprop

RMSprop adapts the learning rate for each parameter using historical gradient magnitudes.

```python
s = np.zeros_like(theta)
epsilon = 1e-8  # To avoid division by zero

for t in range(num_iterations):
    gradient = compute_gradient(X_batch, y_batch, theta)

    s = beta * s + (1 - beta) * gradient**2
    theta -= learning_rate * gradient / (np.sqrt(s) + epsilon)
```

$$\begin{align}
s_t &= \beta s_{t-1} + (1-\beta) (\nabla_\theta J(\theta_t))^2 \\
\theta_{t+1} &= \theta_t - \frac{\alpha}{\sqrt{s_t} + \epsilon} \nabla_\theta J(\theta_t)
\end{align}$$

Parameters with frequent large gradients receive smaller effective learning rates; parameters with small or rare gradients receive larger ones. $\beta = 0.999$ is a typical setting. See [`rmsprop.py`](https://github.com/eima40x4c/Deep-Learning-Reference-Hub/blob/main/src/dlhub/optimizers/rmsprop.py).

## Adam

Adam combines momentum and RMSprop with bias correction.

```python
m = np.zeros_like(theta)  # First moment (momentum)
v = np.zeros_like(theta)  # Second moment (RMSprop)
epsilon = 1e-8
beta1 = 0.9
beta2 = 0.999

for t in range(1, num_iterations + 1):
    gradient = compute_gradient(X_batch, y_batch, theta)
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * gradient**2

    m_corrected = m / (1 - beta1**t)
    v_corrected = v / (1 - beta2**t)

    theta -= learning_rate * m_corrected / (np.sqrt(v_corrected) + epsilon)
```

$$\begin{align}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) \nabla_\theta J(\theta_t) \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) (\nabla_\theta J(\theta_t))^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1-\beta_2^t} \\
\theta_{t+1} &= \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
\end{align}$$

Default hyperparameters are $\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, and $\epsilon = 1e^{-8}$. See [`adam.py`](https://github.com/eima40x4c/Deep-Learning-Reference-Hub/blob/main/src/dlhub/optimizers/adam.py).

## AdamW and AMSGrad

AdamW applies weight decay directly during the parameter update:

```python
# Standard Adam update
theta = theta - learning_rate * m_corrected / (np.sqrt(v_corrected) + epsilon)
# Added weight decay
theta = theta - learning_rate * weight_decay * theta
```

AMSGrad maintains the maximum of past squared gradients instead of only their exponential average. This addresses theoretical convergence issues in Adam.

## Implementation Reference

The generated [optimizer API reference](api/optimizers.md) documents the package implementation. A side-by-side comparison is available in [`comparison.py`](https://github.com/eima40x4c/Deep-Learning-Reference-Hub/blob/main/src/dlhub/optimizers/comparison.py).
