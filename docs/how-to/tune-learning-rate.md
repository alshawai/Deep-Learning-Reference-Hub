# Tune a learning rate

Use a learning-rate finder or range test before committing resources to a full
hyperparameter search.

## 1. Choose the initial range

Start with a range from $10^{-5}$ to $10^{-1}$.

## 2. Run a learning-rate range test

The repository's implementation is documented in the generated
[tuning API reference](../reference/api/tuning.md). Use its learning-rate finder
to run the range test.

## 3. Add warmup and decay

Use cosine annealing or linear warmup followed by decay. Learning-rate schedulers
and warmup strategies can greatly affect training speed and accuracy.

See [learning-rate schedules](../reference/learning-rate-schedules.md) for the
available schedules and their parameters. Repository implementations are also
listed in the generated [optimizer API reference](../reference/api/optimizers.md).

Then continue with [a complete hyperparameter search](run-hyperparameter-search.md).
