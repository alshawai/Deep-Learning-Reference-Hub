# Why parameter initialization matters

Parameter initialization is critical for successful deep learning training. Poor initialization can lead to:

- **Vanishing gradients**: Gradients become exponentially small in deep networks
- **Exploding gradients**: Gradients become exponentially large
- **Symmetry breaking**: All neurons learn the same features
- **Slow convergence**: Training takes much longer to reach optimal solutions

## The mathematical foundation

For proper signal propagation through deep networks, we need to maintain the variance of activations and gradients across layers. The key insight is that _the variance of a layer's output should be approximately equal to the variance of its input._

**Forward Pass Variance Preservation:**

$$Var(output) ≈ Var(input)$$

**Backward Pass Variance Preservation:**

$$Var(gradient) ≈ Var(upstream\_gradient)$$

## Why the schemes differ

### Zero initialization

**Problem**: All neurons learn identical features due to _perfect symmetry_, essentially learning the same parameters, which makes it no different from a regular _machine learning model_ ($1$ layer, $1$ neuron).

### Random small values

**Problem**: Activations and gradients shrink exponentially in deep networks, which causes _vanishing gradients_.

### Xavier/Glorot initialization

**Mathematical Justification:**

- Maintains unit variance for both forward and backward pass
- Derived assuming linear activations (works well for tanh/sigmoid)

### He initialization

**Mathematical Justification:**

- Accounts for the fact that ReLU kills half the neurons on average
- Factor of 2 compensates for the reduced variance due to ReLU

For the equations and implementations, see the [parameter initialization reference](../reference/parameter-initialization.md). The complete mixed framework snippets are in the [regularization reference](../reference/regularization.md#framework-specific-implementations).
