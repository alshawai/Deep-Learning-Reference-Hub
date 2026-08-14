# Gradient checking reference

For the role of gradient checking, see [How gradient checking works](../explanation/gradient-checking.md). For the procedure, see [Run a gradient check](../how-to/run-a-gradient-check.md).

## Mathematical foundation

**Numerical Gradient (Two-sided difference):**

$$ \frac{\partial J}{\partial \theta} \approx \frac{J(θ + \epsilon) - J(θ - \epsilon)}{2 \epsilon} $$

where $\epsilon$ is a small value (typically $1e^{-7}$).

**Relative Difference (between _analytical_ and _numerical_ gradients):**

$$ \text{Difference} = \frac{||\text{Grad} - \text{Grad}_{approx}||_2} {||\text{Grad}||_2 + ||\text{Grad}_{approx}||_2} $$

## Interpretation of results

**Gradient Check Tolerance:**

- **difference < 1e-7**: Excellent! Your implementation is likely correct
- **1e-7 < difference < 1e-5**: Good. Probably correct, but double-check
- **1e-5 < difference < 1e-3**: Warning. Likely a bug in backpropagation
- **difference > 1e-3**: Error. Definitely a bug in your implementation

## Implementation

A complete reference implementation for **Gradient Checking** is provided in the [training techniques API reference](api/training.md).

[Gradient Checking](api/training.md) - Compares analytical and numerical gradients using a two-sided difference method to ensure backpropagation correctness. Prints the relative difference for debugging.
