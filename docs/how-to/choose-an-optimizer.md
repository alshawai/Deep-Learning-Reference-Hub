# How to Choose an Optimizer

Start with the problem and training stage, then adjust the learning rate, batch size, and regularization based on the observed behavior. The equations and defaults are in [optimizer update rules](../reference/optimizer-update-rules.md); schedule choices are in [learning-rate schedules](../reference/learning-rate-schedules.md).

## Make the Initial Choice

- **Most problems**: Start with Adam or AdamW.
- **Computer vision**: Try SGD with momentum for the final training phase.
- **NLP and transformers**: Use AdamW with cosine annealing.
- **Large-scale training**: Use Adam with gradient clipping.

## Set Starting Values

- **Learning rate**: Start with 0.001 for Adam and 0.01 for SGD.
- **Batch size**: Use 32-256 for most problems, with larger batches for very large datasets.
- **Momentum**: Use 0.9 for SGD and 0.9 for Adam's $\beta_1$.
- **Weight decay**: Use 0.01-0.1 for regularization.

Mini-batch gradient descent is generally the practical default because it balances speed, memory use, vectorization, and gradient stability. See the [mini-batch update rule](../reference/optimizer-update-rules.md#mini-batch-gradient-descent) for batch-size ranges.

## Use a Modern Training Loop

```python
# Modern training loop structure - in PyTorch
optimizer = AdamW(parameters, lr=1e-3, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

for epoch in range(epochs):
    for batch in train_loader:
        loss = model(batch)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )  # Gradient clipping - optional

        optimizer.step()
    scheduler.step()
```

## Use Framework Implementations

PyTorch:

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

TensorFlow/Keras:

```python
import tensorflow as tf

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
```

## Diagnose Training Behavior

- **Convergence issues**: Check the learning rate and add gradient clipping.
- **Poor generalization**: Add weight decay and reduce the learning rate.
- **Need for a stable default**: Use Adam or AdamW.
- **Need for better final vision-model performance**: Try SGD with momentum in the final training phase.

For complete runnable examples, see the [`comparison.py`](https://github.com/eima40x4c/Deep-Learning-Reference-Hub/blob/main/src/dlhub/optimizers/comparison.py) implementation and the generated [optimizer API reference](../reference/api/optimizers.md).
