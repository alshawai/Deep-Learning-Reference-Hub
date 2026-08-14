# Network Shapes and Dimensions

See [Forward and Backward Propagation](../explanation/forward-and-backward-propagation.md) for how these tensors are used.

## Network Architecture and Notation

### Network Structure

- Input: $X \in \mathbb{R}^{n^{[0]} \times m}$ where:
    - $n^{[0]}$ = number of features
    - $m$ = number of examples
- Layers: $L$ layers total (including output layer)
- Layer $l$ has $n^{[l]}$ neurons for $l = 1, 2, \ldots, L$
- Parameters:
    - $W^{[l]} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}$ (weight matrix for layer $l$)
    - $b^{[l]} \in \mathbb{R}^{n^{[l]} \times 1}$ (bias vector for layer $l$)
- Activations:
    - $A^{[l]} \in \mathbb{R}^{n^{[l]} \times m}$ (activation matrix for layer $l$)
    - $A^{[0]} = X$ (input layer)

## Dimensional Analysis

For layer $l$ with $n^{[l]}$ neurons and $n^{[l-1]}$ neurons in the previous layer:

### Forward Propagation Dimensions

- $Z^{[l]} \in \mathbb{R}^{n^{[l]} \times m}$
- $W^{[l]} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}$
- $A^{[l-1]} \in \mathbb{R}^{n^{[l-1]} \times m}$
- $b^{[l]} \in \mathbb{R}^{n^{[l]} \times 1}$

**Verification:**

$$W^{[l]} A^{[l-1]} + b^{[l]} \rightarrow (n^{[l]} \times n^{[l-1]}) \cdot (n^{[l-1]} \times m) + (n^{[l]} \times 1) = (n^{[l]} \times m)$$

### Backward Propagation Dimensions

- $dZ^{[l]} \in \mathbb{R}^{n^{[l]} \times m}$
- $dW^{[l]} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}$ (same as $W^{[l]}$)
- $db^{[l]} \in \mathbb{R}^{n^{[l]} \times 1}$ (same as $b^{[l]}$)
- $dA^{[l-1]} \in \mathbb{R}^{n^{[l-1]} \times m}$ (same as $A^{[l-1]}$)

**Verification:**

$$dZ^{[l]} (A^{[l-1]})^T \rightarrow (n^{[l]} \times m) \cdot (m \times n^{[l-1]}) = (n^{[l]} \times n^{[l-1]})$$
$$(W^{[l]})^T dZ^{[l]} \rightarrow (n^{[l-1]} \times n^{[l]}) \cdot (n^{[l]} \times m) = (n^{[l-1]} \times m)$$
