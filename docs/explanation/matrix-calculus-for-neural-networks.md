# Matrix Calculus for Neural Networks

These matrix rules support the derivation in [Forward and Backward Propagation](forward-and-backward-propagation.md).

## Key Matrix Derivatives

For $Z = WA + b$:

$$\frac{\partial Z}{\partial W} = A^T, \quad \frac{\partial Z}{\partial A} = W^T, \quad \frac{\partial Z}{\partial b} = \mathbf{1}$$

## Chain Rule in Matrix Form

$$\frac{\partial J}{\partial W^{[l]}} = \frac{\partial J}{\partial Z^{[l]}} \frac{\partial Z^{[l]}}{\partial W^{[l]}} = dZ^{[l]} (A^{[l-1]})^T$$

## Vectorization Benefits

- Process all $m$ training examples simultaneously
- Efficient GPU computation through BLAS operations
- Reduced computational complexity from $O(mn^2L)$ to $O(n^2L)$ per iteration
