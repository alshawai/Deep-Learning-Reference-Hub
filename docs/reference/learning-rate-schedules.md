# Learning-Rate Schedules

Learning-rate scheduling adapts the learning rate during training. This page lists the schedule formulas and compact implementations. See [optimizer update rules](optimizer-update-rules.md) for the optimizer equations and [how to choose an optimizer](../how-to/choose-an-optimizer.md) for a training strategy.

## Why Scheduling Matters

- **Early training**: A higher learning rate enables faster progress.
- **Later training**: A lower learning rate supports fine-tuning.
- **Plateau detection**: A stalled validation loss can trigger a reduction.

## Step Decay

Reduce the learning rate by a factor at specific epochs:

```python
def step_decay(epoch, lr):
    if epoch in [30, 60, 90]:
        return lr * 0.1
    return lr
```

## Exponential Decay

Gradually reduce the learning rate:

```python
def exponential_decay(epoch, lr):
    return lr * np.exp(-0.1 * epoch)
```

## Cosine Annealing

Smoothly reduce the learning rate following a cosine curve:

```python
def cosine_annealing(epoch, lr, T_max):
    return lr * (1 + np.cos(np.pi * epoch / T_max)) / 2
```

## Reduce on Plateau

Reduce the learning rate when validation loss stops improving:

```python
def reduce_on_plateau(val_loss, lr, patience=10, factor=0.5):
    if no_improvement_for_patience_epochs:  # Core logic needs to be handled.
        return lr * factor
    return lr
```

The schedule implementations are collected in [`schedules.py`](https://github.com/eima40x4c/Deep-Learning-Reference-Hub/blob/main/src/dlhub/optimizers/schedules.py).
