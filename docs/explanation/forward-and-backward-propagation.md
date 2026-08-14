# Forward and Backward Propagation

See [Network Shapes and Dimensions](../reference/network-shapes-and-dimensions.md) for the notation and tensor shapes used here, [Activation Functions](../reference/activation-functions.md) for activation derivatives, and [Matrix Calculus for Neural Networks](matrix-calculus-for-neural-networks.md) for the underlying matrix rules.

## Forward Propagation

For each layer $l = 1, 2, \ldots, L$:
**Linear Transformation:** $Z^{[l]}=W^{[l]}A^{[l−1]}+b{[l]}$
**Activation Function:** $A^{[l]}=g^{[l]}(Z{[l]})$
Where $g^{[l]}$ is the activation function for layer $l$.

### Complete Forward Pass

$$\begin{align}
Z^{[1]} &= W^{[1]} X + b^{[1]} &\quad  A^{[1]} &= g^{[1]}(Z^{[1]}) \\
Z^{[2]} &= W^{[2]} A^{[1]} + b^{[2]} &\quad A^{[2]} &= g^{[2]}(Z^{[2]}) \\
&\vdots & &\vdots \\
Z^{[L]} &= W^{[L]} A^{[L-1]} + b^{[L]} &\quad A^{[L]} &= g^{[L]}(Z^{[L]}) \quad \text{(Final output)}
\end{align}$$

The final prediction is $\hat{Y} = A^{[L]}$.

## Cost Function

For **binary cross-entropy** (logistic regression output):

$$J = -\frac{1}{m} \sum_{i=1}^{m}~[ Y^{(i)} \log(A^{[L] (i)}) + (1-Y^{(i)}) \log(1-A^{[L] (i)})]$$

For **mean squared error**:

$$J = \frac{1}{2m} \sum_{i=1}^{m} ||A^{[L] (i)} - Y^{(i)}||^2$$

## Backward Propagation: Complete Mathematical Derivation

### Step 1: Derivative with respect to Output Layer Activations

For **binary cross-entropy**:

$$\frac{\partial J}{\partial A^{[L]}} = -\frac{1}{m} \left[ \frac{Y}{A^{[L]}} - \frac{1-Y}{1-A^{[L]}} \right]$$

For **mean squared error**:

$$\frac{\partial J}{\partial A^{[L]}} = \frac{1}{m} (A^{[L]} - Y)$$

### Step 2: Derivative with respect to Output Layer Pre-activations

Using the chain rule:

$$\frac{\partial J}{\partial Z^{[L]}} = \frac{\partial J}{\partial A^{[L]}} \cdot \frac{\partial A^{[L]}}{\partial Z^{[L]}}$$

Since $A^{[L]} = g^{[L]}(Z^{[L]})$:

$$\frac{\partial A^{[L]}}{\partial Z^{[L]}} = g'^{[L]}(Z^{[L]})$$

Therefore:

$$dZ^{[L]} = \frac{\partial J}{\partial Z^{[L]}} = \frac{\partial J}{\partial A^{[L]}} \odot g'^{[L]}(Z^{[L]})$$

**Special case for sigmoid + cross-entropy:**
When $g^{[L]}(z) = \sigma(z) = \frac{1}{1+e^{-z}}$ and using cross-entropy:

$$dZ^{[L]} = A^{[L]} - Y$$

### Step 3: Derivatives with respect to Parameters of Layer L

**Weight derivatives:**

$$\frac{\partial J}{\partial W^{[L]}} = \frac{\partial J}{\partial Z^{[L]}} \cdot \frac{\partial Z^{[L]}}{\partial W^{[L]}}$$

Since $Z^{[L]} = W^{[L]} A^{[L-1]} + b^{[L]}$:

$$\frac{\partial Z^{[L]}}{\partial W^{[L]}} = A^{[L-1]T}$$

Therefore:

$$dW^{[L]} = \frac{1}{m} dZ^{[L]} (A^{[L-1]})^T$$

**Bias derivatives:**

$$\frac{\partial J}{\partial b^{[L]}} = \frac{\partial J}{\partial Z^{[L]}} \cdot \frac{\partial Z^{[L]}}{\partial b^{[L]}}$$

Since $\frac{\partial Z^{[L]}}{\partial b^{[L]}} = \mathbf{1}$ (broadcasting):

$$db^{[L]} = \frac{1}{m} \text{sum}(dZ^{[L]}, \text{axis}=1, \text{keepdims}=\text{True})$$

### Step 4: Derivative with respect to Previous Layer Activations

$$\frac{\partial J}{\partial A^{[L-1]}} = \frac{\partial J}{\partial Z^{[L]}} \cdot \frac{\partial Z^{[L]}}{\partial A^{[L-1]}}$$

Since $Z^{[L]} = W^{[L]} A^{[L-1]} + b^{[L]}$:

$$\frac{\partial Z^{[L]}}{\partial A^{[L-1]}} = (W^{[L]})^T$$

Therefore:

$$dA^{[L-1]} = (W^{[L]})^T dZ^{[L]}$$

### Step 5: General Recursive Formula for Hidden Layers

For any layer $l$ where $1 \leq l < L$:

**Pre-activation derivatives:**

$$dZ^{[l]} = dA^{[l]} \odot g'^{[l]}(Z^{[l]})$$

**Weight derivatives:**

$$dW^{[l]} = \frac{1}{m} dZ^{[l]} (A^{[l-1]})^T$$

**Bias derivatives:**

$$db^{[l]} = \frac{1}{m} \text{sum}(dZ^{[l]}, \text{axis}=1, \text{keepdims}=\text{True})$$

**Previous layer activation derivatives:**

$$dA^{[l-1]} = (W^{[l]})^T dZ^{[l]}$$

## Complete Backward Propagation Algorithm

### Mathematical Formulation

$\text{Output Layer:}$

$$\begin{align}
dZ^{[L]} &= \frac{\partial J}{\partial A^{[L]}} \odot g'^{[L]}(Z^{[L]}) \\
dW^{[L]} &= \frac{1}{m} dZ^{[L]} (A^{[L-1]})^T \\
db^{[L]} &= \frac{1}{m} \sum_{i=1}^{m} dZ^{[L] (\cdot,i)} \\
dA^{[L-1]} &= (W^{[L]})^T dZ^{[L]} \\
\end{align}$$

$\text{Hidden Layers } ( L-1 \geq l \geq 1)\text{: }$

$$\begin{align}
dZ^{[l]} &= dA^{[l]} \odot g'^{[l]}(Z^{[l]}) \\
dW^{[l]} &= \frac{1}{m} dZ^{[l]} (A^{[l-1]})^T \\
db^{[l]} &= \frac{1}{m} \sum_{i=1}^{m} dZ^{[l] (\cdot,i)} \\
dA^{[l-1]} &= (W^{[l]})^T dZ^{[l]} \quad \text{(if } l > 1\text{)}
\end{align}$$

After computing the gradients, apply the [Parameter Update Rule](../reference/parameter-update-rule.md).
