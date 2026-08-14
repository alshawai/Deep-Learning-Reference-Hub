# Run a hyperparameter search

Use this workflow to define, execute, and evaluate a deep-learning
hyperparameter search without exhausting the budget on weak configurations or
overfitting the validation set.

## 1. Establish a baseline

Confirm that one known-good configuration trains end to end and record its
metric, runtime, memory use, seed, and software environment. For example:

```python
config = {
    "learning_rate": 3e-4,
    "batch_size": 32,
    "weight_decay": 1e-4,
    "dropout": 0.1,
    "warmup_steps": 1000,
}
```

Treat values as starting points, not universal defaults. Typical domain-specific
starting configurations include:

```python
vision_defaults = {
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "augmentation_strength": 0.2,
    "mixup_alpha": 0.2,
}

nlp_defaults = {
    "learning_rate": 5e-5,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "label_smoothing": 0.1,
    "attention_dropout": 0.1,
}

llm_defaults = {
    "learning_rate": 1e-4,
    "batch_size": 2048,
    "gradient_accumulation": 8,
    "beta2": 0.95,
    "weight_decay": 0.1,
}
```

For a new project, a practical cadence is to establish the baseline on day one,
tune the learning rate over days two and three, scale the architecture over days
four and five, and reserve Bayesian or multi-objective optimization for the
second week and beyond. Adjust that schedule to the cost of each training run.

## 2. Define the objective and data splits

Choose the primary metric before running trials. For deployment-sensitive work,
record secondary objectives such as latency, model size, or fairness and retain
the Pareto-optimal configurations rather than collapsing incompatible goals
prematurely.

Repeated validation-set evaluation can overfit the search itself. Use multiple
validation splits or cross-validation when feasible, reserve a separate test set
for final evaluation, and limit the total number of tuning decisions made from
the same validation data.

## 3. Design the search space

Start with the high-impact hyperparameters in the
[search-method reference](../reference/hyperparameter-search-methods.md#hyperparameters-by-impact).
Tune the learning rate first with the
[learning-rate range procedure](tune-learning-rate.md), then scale architecture
to data complexity and use established scaling laws when available.

Use log-scaled distributions for multiplicative quantities such as learning rate
and weight decay. Use categorical distributions for valid discrete choices and
integer distributions for counts. Begin with broad, defensible ranges, inspect
the first results, and narrow the space only when the evidence supports it.

## 4. Match the method to the budget

With limited compute:

1. Run random search for approximately 20-50 trials.
2. Apply early stopping aggressively.
3. Focus on Tier 1 hyperparameters.
4. Use smaller models for initial exploration, then verify at target scale.

With abundant compute:

1. Use Bayesian optimization for expensive fine-tuning.
2. Use multi-fidelity methods to allocate training budget adaptively.
3. Consider population-based training for long runs.
4. Use neural architecture search only when architecture is genuinely part of
   the objective.

The [search-method catalog](../reference/hyperparameter-search-methods.md#search-methods)
details each method. The generated [tuning API reference](../reference/api/tuning.md)
documents the repository's implementations.

## 5. Execute reproducible trials

Record every sampled configuration, random seed, score, stopping reason, and
resource budget. Parallelize independent trials across GPUs or machines, and use
early stopping or multi-fidelity allocation instead of fully training clearly
poor configurations.

Do not assume that a promising small model or data subset transfers perfectly.
Re-evaluate finalists at the target model size, dataset size, input resolution,
and training duration.

## 6. Compare finalists across seeds

Single runs contain random variation. Re-run finalists with several seeds and
report the distribution of results. A simple two-sample test can flag whether
an observed difference is distinguishable from run-to-run variation:

```python
import scipy.stats as stats


def compare_configurations(config_a_scores, config_b_scores):
    """Compare two configurations with an independent two-sample t-test."""
    statistic, p_value = stats.ttest_ind(config_a_scores, config_b_scores)

    if p_value < 0.05:
        return "Statistically significant difference"
    return "No significant difference"


config_a_scores = [0.92, 0.91, 0.93, 0.90, 0.92]
config_b_scores = [0.89, 0.90, 0.88, 0.91, 0.89]
result = compare_configurations(config_a_scores, config_b_scores)
```

Select the statistical test and significance threshold before inspecting the
comparison. Statistical significance alone does not establish practical
importance, so report the effect size and uncertainty as well.

## 7. Finalize the configuration

Choose a configuration using the predefined objective and deployment
constraints. Evaluate it once on the held-out test set, and retain the complete
search history for reproducibility and later hyperparameter-importance analysis.

For the reasoning behind prioritization, resource allocation, transfer, and
multi-objective search, see
[the hyperparameter tuning landscape](../explanation/hyperparameter-tuning-landscape.md).
