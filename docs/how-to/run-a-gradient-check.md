# Run a gradient check

Compute the analytical gradients with backpropagation, then pass them with the
model parameters, input data, labels, and cost function to `gradient_check`. The
function computes numerical gradients and returns their relative difference from
the supplied analytical gradients. See the
[training techniques API reference](../reference/api/training.md) for the exact
interface, then interpret the result using the
[gradient check tolerances](../reference/gradient-checking.md#interpretation-of-results).

## Common issues and debugging tips

1. **Regularization**: Don't forget to include regularization terms in both forward and backward pass
2. **Dropout**: Turn off dropout during gradient checking
3. **Batch Normalization**: Use the same batch for both forward passes
4. **Numerical Precision**: Use double precision (float64) for gradient checking
5. **Random Initialization**: Use fixed random seed for reproducibility

## Development practice

- Essential during development
- Use only on small subsets of data
- Disable all stochastic elements (dropout, batch norm in training mode)

Always implement gradient checking first, then optimize for performance.

For why the comparison works, see [How gradient checking works](../explanation/gradient-checking.md).
