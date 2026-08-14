# Parameter initialization reference

For the reasoning behind these schemes, see [Why parameter initialization matters](../explanation/parameter-initialization.md). Complete mixed initialization and regularization examples are in the [regularization reference](regularization.md#framework-specific-implementations).

## Common initialization methods

### 1. Zero initialization

```python
W = np.zeros((n_in, n_out))
```

### 2. Random small values

```python
W = np.random.randn(n_in, n_out) * 0.01
```

### 3. Xavier/Glorot initialization

**Best for**: Tanh and Sigmoid activation functions

**Normal Distribution:**

```python
W = np.random.randn(n_in, n_out) * np.sqrt(1.0 / n_in)
```

**Uniform Distribution:**

```python
limit = np.sqrt(6.0 / (n_in + n_out))
W = np.random.uniform(-limit, limit, (n_in, n_out))
```

### 4. He initialization

**Best for**: ReLU and its variants

**Normal Distribution:**

```python
W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
```

**Uniform Distribution:**

```python
limit = np.sqrt(6.0 / n_in)
W = np.random.uniform(-limit, limit, (n_in, n_out))
```

### 5. Modern initialization strategies

**LeCun Initialization** (for SELU):

```python
W = np.random.randn(n_in, n_out) * np.sqrt(1.0 / n_in)
```

**Orthogonal Initialization** (for RNNs):

```python
# Uses orthogonal matrices to prevent vanishing/exploding gradients
W = orthogonal_matrix(n_in, n_out)
```

## Bias initialization

**Standard Practice:**

```python
b = np.zeros(n_out)  # Initialize biases to zero
```

**Exception for ReLU:**

```python
b = np.full(n_out, 0.01)  # Small positive bias to ensure initial activation
```

## 2024 recommendations

**Initialization Strategy:**

- **CNNs**: He initialization with ReLU activations
- **Transformers**: Xavier initialization with layer normalization
- **RNNs**: Orthogonal initialization for recurrent connections

**Parameter Initialization**: Use He initialization for ReLU networks, Xavier for tanh/sigmoid networks
