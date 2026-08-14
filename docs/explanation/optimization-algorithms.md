# Optimization Algorithms in Deep Learning

Optimization algorithms are the backbone of deep learning training. They determine how quickly and effectively a model converges to useful parameters. Poor optimization can lead to:

- **Slow convergence**: Training takes unnecessarily long.
- **Poor final performance**: The model gets stuck in suboptimal solutions.
- **Training instability**: Loss oscillates or diverges during training.
- **Inefficient resource usage**: Computational power and time are wasted.

For the equations and defaults in lookup form, see [optimizer update rules](../reference/optimizer-update-rules.md). For a practical selection sequence, see [how to choose an optimizer](../how-to/choose-an-optimizer.md).

## The Optimization Landscape

Deep learning optimization involves navigating a high-dimensional, non-convex loss landscape. Numerous challenges arise:

- **Saddle points**: Points where the gradient is zero but the point is not optimal.
- **Local minima**: Suboptimal solutions that can trap basic gradient descent.
- **Vanishing/exploding gradients**: Gradients become too small or too large.
- **Ill-conditioned problems**: Different dimensions have vastly different curvatures.

Given a loss function $J(\theta)$ where $\theta$ represents model parameters, optimization seeks to find:

$$\theta^* = \arg \min_\theta J(\theta)$$

The basic iterative update is:

$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta J(\theta_t)$$

where $\alpha$ is the learning rate and $\nabla_\theta J(\theta_t)$ is the gradient with respect to $\theta$.

## Exponential Weighted Averages

Exponential weighted averages (EWA), or exponentially weighted moving averages, provide a way to compute running averages that give more weight to recent values. They are the shared mechanism behind momentum, RMSprop, and Adam.

The formula is:

$$v_t = \beta v_{t-1} + (1-\beta) \theta_t$$

where:

- $v_t$ is the exponentially weighted average at time $t$.
- $\beta$ is the momentum parameter, typically 0.9-0.999.
- $\theta_t$ is the current value of parameter $\theta$.
- $v_0 = 0$ is the initialization.

Intuitively, $v_t$ approximates the average of the last $\frac{1}{1-\beta}$ values:

- $\beta = 0.9$ is approximately the average of the last 10 values.
- $\beta = 0.99$ is approximately the average of the last 100 values.
- $\beta = 0.999$ is approximately the average of the last 1000 values.

### Bias Correction

Early estimates are biased toward zero because $v_0 = 0$. Bias correction addresses this with:

$$v_t^{corrected} = \frac{v_t}{1 - \beta^t}$$

At $t=1$:

$$v_1^{corrected} = \frac{(1-\beta)\theta_1}{1-\beta} = \theta_1$$

As $t \to \infty$, $\beta^t \to 0$ when $\beta < 1$, so $v_t^{corrected} \to v_t$. The implementation is available in [`exponential_weighted_averages.py`](https://github.com/eima40x4c/Deep-Learning-Reference-Hub/blob/main/src/dlhub/optimizers/exponential_weighted_averages.py).

```python
def exponential_weighted_average(values, beta=0.9, bias_correction=True):
    v = 0
    corrected_values = []

    for t, value in enumerate(values, 1):
        v = beta * v + (1 - beta) * value

        if bias_correction:
            v_corrected = v / (1 - beta**t)
            corrected_values.append(v_corrected)
        else:
            corrected_values.append(v)

    return corrected_values
```

## What the Methods Trade

Batch gradient descent has smooth, stable gradient estimates and is guaranteed to converge for convex functions with a proper learning rate, but it is computationally expensive, memory intensive, and slow for large datasets.

Stochastic gradient descent updates after one example. It is fast, memory efficient, can escape poor local solutions through noise, and supports online learning. Its gradient estimates are noisy, so it can oscillate and converge slowly near an optimum.

Mini-batch gradient descent combines these properties. It is vectorization friendly, uses manageable memory, is more stable than pure SGD, and is faster than full-batch updates. Larger batches provide better gradient estimates, but very large batches may hurt generalization through the large-batch trap.

Momentum is analogous to a ball rolling down a hill: the gradient supplies the current slope direction, momentum is the velocity accumulated from previous movements, and the ball accelerates in consistent directions while damping oscillations. This produces faster convergence, helps cross small barriers, and smooths noisy gradient estimates.

RMSprop adapts the step for each parameter. Automatic scaling avoids manually setting a separate learning rate for every parameter, reduces tuning, and is less sensitive to the initial learning-rate choice. It is particularly useful for sparse or non-stationary problems, although accumulated squared gradients can cause learning-rate decay.

Adam combines momentum and adaptive per-parameter learning rates with bias correction. It often converges quickly, is comparatively insensitive to hyperparameter choices, handles sparse gradients well, and is a robust default across diverse problems, including NLP and sparse-data tasks.

AdamW separates weight decay from Adam's adaptive gradient update. This avoids the dilution of the regularization effect caused by applying L2 regularization inside Adam's adaptive scaling. AMSGrad addresses theoretical convergence concerns in Adam by retaining the maximum of past squared gradients rather than only an exponential average.

The complete update equations and method-specific characteristics are in [optimizer update rules](../reference/optimizer-update-rules.md), while schedule formulas are in [learning-rate schedules](../reference/learning-rate-schedules.md).
