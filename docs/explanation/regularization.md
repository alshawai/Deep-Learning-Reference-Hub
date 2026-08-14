# Regularization techniques

Regularization prevents overfitting by adding constraints or noise to the learning process, helping models generalize better to unseen data.

## L2 regularization

**Most Common**: Penalizes large weights by adding their squared magnitude to the loss.

**Benefits:**

- Prevents weights from becoming too large
- Encourages weight sharing
- Smooth decision boundaries

## L1 regularization

**Sparsity-Inducing**: Promotes sparse weights (many weights become exactly zero).

**Benefits:**

- Automatic feature selection
- Sparse models (smaller memory footprint)
- Interpretable models

## Dropout

**Core Idea**: Randomly set a fraction of neurons to zero during training, forcing the network to learn redundant representations.

The division by `keep_prob` is called **inverted dropout** and it's crucial for maintaining the expected value of activations:

> **Without Inverted Dropout:**
>
> - **Training:** `E[X_dropout] = keep_prob * E[X]` (scaled down)
> - **Testing:** `E[X_test] = E[X]` (original scale)
> - **Problem:** _Different scales_ between training and testing!
>
> **With Inverted Dropout:**
>
> - **Training:** `E[X_dropout] = E[X * mask / keep_prob] = E[X] * E[mask] / keep_prob = E[X] * keep_prob / keep_prob = E[X]`
> - **Testing:** `E[X_test] = E[X]` (no dropout applied)
> - **Benefit:** _Same expected scale_ in both training and testing!

**Modern Standard:**

The inverted dropout (scaling during training) is now the standard approach because:

1. **No inference overhead:** No need to scale during testing
2. **Cleaner implementation:** Test time is just forward pass without modifications
3. **Framework compatibility:** All major frameworks (PyTorch, TensorFlow) use this approach

**Benefits:**

- Reduces overfitting significantly
- Improves generalization
- Acts as ensemble method (averaging multiple sub-networks)

**Modern Considerations:**

- Often not needed with batch normalization
- Can increase training time
- Less effective in very deep networks with proper normalization

## Batch normalization

**Revolutionary Technique**: Normalizes _inputs to each layer_, dramatically improving training stability and speed.

**Benefits:**

- **Accelerates training**: Often 2-10x faster convergence
- **Reduces sensitivity to initialization**: Can use higher learning rates
- **Regularization effect**: Reduces need for dropout
- **Gradient flow**: Helps with vanishing gradient problem

## Early stopping

**Simple yet Effective**: Stop training when validation performance starts degrading.

**Benefits:**

- Prevents overfitting without hyperparameter tuning
- Computationally efficient
- Works with any model architecture

## Modern practice

**Regularization Hierarchy:**

- **First choice**: Batch/Layer Normalization
- **Second choice**: Weight decay (L2 regularization)
- **Third choice**: Dropout (if needed)
- **Always**: Early stopping

**Regularization**: Batch normalization is often sufficient; add L2 regularization and dropout as needed

**Modern Practice**: Combine techniques thoughtfully - batch normalization often reduces need for dropout

For equations, implementations, and framework examples, see the [regularization reference](../reference/regularization.md).
