# Regularization reference

For the rationale behind these techniques, see [Regularization techniques](../explanation/regularization.md).

## L2 regularization (weight decay)

**Mathematical Form:**

$$ J_{regularized} = J_{original} + \frac{\lambda}{2m} ~ \sum^m W^2 $$

**Implementation:**

```python
# In loss function
l2_penalty = 0.5 * lambda_reg * np.sum(W**2)
total_loss = original_loss + l2_penalty

# In gradient computation
dW += lambda_reg * W
```

## L1 regularization (Lasso)

**Mathematical Form:**

$$ J_{regularized} = J_{original} + \frac{\lambda}{m} ~ \sum^m \left| W \right| $$

**Implementation:**

```python
# In loss function
l1_penalty = lambda_reg * np.sum(np.abs(W))
total_loss = original_loss + l1_penalty

# In gradient computation
dW += lambda_reg * np.sign(W)
```

## Dropout

**Mathematical Formulation:**

During training:

```python
h_dropout = h * mask / keep_prob
```

where $\text{mask} \approx \text{Bernoulli(keep-prob)} $

During inference:

```python
h_inference = h  # No dropout, but scaled during training
```

**Implementation:**

```python
def dropout_forward(X, keep_prob, training=True):
    if training:
        mask = np.random.binomial(1, keep_prob, X.shape) / keep_prob
        return X * mask, mask
    else:
        return X, None


def dropout_backward(dout, mask):
    return dout * mask
```

**Key Parameters:**

- **keep_prob**: Probability of keeping a neuron (typically 0.5-0.8)
- **Inverted dropout**: Scale activations during training (modern approach)

## Batch normalization

**Mathematical Formulation:**

$$ \begin{align}
\text{Batch mean: } \quad & \mu_B = \frac 1 m \times \sum^m X \\
\text{Batch variance: } \quad & \sigma^2_B = \frac 1 m \times \sum^m~(X - \mu_B)^2 \\
\text{Normalize: } \quad & \hat X = \frac{X - \mu_B} {\sqrt{\sigma^2_B + \epsilon}} \\
\text{Scale and shift: } \quad & Y = \gamma \times \hat X + \beta
\end{align} $$

**Implementation:**

```python
def batch_norm_forward(X, gamma, beta, eps=1e-8):
    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0)
    X_norm = (X - mu) / np.sqrt(var + eps)
    out = gamma * X_norm + beta

    # Cache for backward pass
    cache = (X, X_norm, mu, var, gamma, beta, eps)
    return out, cache
```

**Modern Usage:**

- Standard in most CNN architectures
- Applied after linear transformation, before activation
- Learnable parameters γ (scale) and β (shift)

## Early stopping

A complete reference implementation for **Early Stopping** is provided in the [training techniques API reference](api/training.md).

[Early Stopping](api/training.md) - Implements a validation-based early stopping strategy, halting training when the validation loss stops improving. Includes configurable patience and verbose logging.

## Framework-specific implementations

These complete snippets intentionally keep initialization and regularization together. See the [parameter initialization reference](parameter-initialization.md) for the individual initialization schemes.

**PyTorch:**

```python
import torch.nn as nn

# He initialization
nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")

# Xavier initialization
nn.init.xavier_uniform_(layer.weight)

# Regularization
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 10),
)
```

**TensorFlow/Keras:**

```python
from tensorflow.keras import layers, initializers

model = tf.keras.Sequential(
    [
        layers.Dense(
            256,
            activation="relu",
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            use_bias=False,
        ),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)
```

## Complete implementation

[Complete Example: Deep Neural Network with All Techniques](api/nn.md) - An end-to-end neural network implementation integrating He initialization, dropout, batch normalization, L2 regularization, and early stopping.
