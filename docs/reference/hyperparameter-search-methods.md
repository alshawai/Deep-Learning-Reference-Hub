# Hyperparameter search methods

This reference catalogs high-impact deep-learning hyperparameters, common
search methods, and the distributions and defaults used to define search
spaces. For selection guidance, see
[how to run a hyperparameter search](../how-to/run-hyperparameter-search.md).

## Hyperparameters by impact

### Tier 1: critical hyperparameters

Tune these first because they usually have the largest effect on performance.

| Hyperparameter | Typical values | Notes |
| --- | --- | --- |
| Learning rate | $10^{-5}$ to $10^{-1}$ | Transformers often require lower rates than CNNs. Use the [learning-rate tuning guide](../how-to/tune-learning-rate.md). |
| Architecture depth | Number of layers | Controls capacity and optimization depth. |
| Architecture width | Hidden units per layer | Controls capacity and compute. |
| Attention heads | Architecture-dependent | Applies to transformer models. |
| Batch size | Powers of two, commonly 16-512 | Batches above 1024 generally require large-batch training techniques. |

### Tier 2: important hyperparameters

| Category | Hyperparameter | Typical values |
| --- | --- | --- |
| Optimization | Adam $\beta_1$ | $[0.9, 0.99]$ |
| Optimization | Adam $\beta_2$ | $[0.99, 0.999]$ |
| Optimization | Weight decay $\lambda$ | $[10^{-6}, 10^{-2}]$ |
| Regularization | Dropout rate $p$ | $[0.1, 0.5]$ |
| Regularization | Label smoothing $\epsilon$ | $[0.05, 0.2]$ |

### Tier 3: fine-tuning hyperparameters

- **Numerical stability:** epsilon values for batch normalization and layer
  normalization, and gradient-clipping thresholds.
- **Training dynamics:** warmup steps and decay factors for learning-rate
  schedules.

## Transformer-specific hyperparameters

| Category | Hyperparameter | Typical values or choices |
| --- | --- | --- |
| Attention | Number of heads | 8, 12, or 16 |
| Attention | Head dimension | 64 or 128 |
| Attention | Attention dropout | Tuned separately from regular dropout |
| Position encoding | Maximum sequence length | The required context-window size |
| Position encoding | Embedding type | Learned or sinusoidal |
| Layer configuration | Feed-forward ratio | Commonly 4 times the model dimension |
| Layer configuration | Normalization placement | Pre-layer or post-layer normalization |

## Search methods

### Manual search

Manual search starts from known defaults, changes one hyperparameter at a time,
and uses domain knowledge to interpret the result.

**Advantages:** it builds understanding, incorporates domain expertise, and is
useful during initial exploration.

**Limitations:** it is time-intensive, does not scale, can miss interactions,
and is vulnerable to human bias and local optima.

### Grid search

For hyperparameters $\theta_1, \theta_2, \ldots, \theta_k$ with discrete sets of
values, grid search evaluates the Cartesian product:

$$
\Theta = \{\theta_1^{(1)}, \theta_1^{(2)}, \ldots\}
\times \{\theta_2^{(1)}, \theta_2^{(2)}, \ldots\}
\times \cdots
\times \{\theta_k^{(1)}, \theta_k^{(2)}, \ldots\}.
$$

```python
learning_rates = [0.1, 0.01, 0.001]
batch_sizes = [32, 64, 128]
hidden_units = [128, 256, 512]

# Total combinations: 3 x 3 x 3 = 27 experiments
for lr in learning_rates:
    for bs in batch_sizes:
        for hu in hidden_units:
            model = train_model(lr=lr, batch_size=bs, hidden_units=hu)
            evaluate_model(model)
```

**Advantages:** systematic, reproducible, exhaustive within the grid, and
parallelizable.

**Limitations:** exponential growth in the number of dimensions, wasted trials
in poor regions, and no evaluation of values between grid points.

### Random search

Random search samples each hyperparameter from a probability distribution:

$$
\theta_i \sim P_i(\theta_i), \qquad i = 1, 2, \ldots, k.
$$

```python
import numpy as np
from scipy.stats import loguniform, uniform


def sample_hyperparameters():
    return {
        "learning_rate": loguniform.rvs(1e-5, 1e-1),
        "batch_size": np.random.choice([16, 32, 64, 128, 256]),
        "hidden_units": np.random.randint(64, 512),
        "dropout_rate": uniform.rvs(0.1, 0.4),
        "weight_decay": loguniform.rvs(1e-6, 1e-2),
    }


configurations = [sample_hyperparameters() for _ in range(100)]
```

Use log-uniform distributions for learning rates and regularization parameters,
uniform sampling from valid discrete choices, and uniform or integer
distributions for architecture parameters.

Random search explores more unique values per dimension than grid search and is
effective when only a few hyperparameters strongly affect the objective. See
[the hyperparameter tuning landscape](../explanation/hyperparameter-tuning-landscape.md#why-random-search-often-beats-grid-search)
for the reasoning.

### Bayesian optimization

Bayesian optimization builds a probabilistic surrogate for the objective
$f(\theta)$ and uses it to select promising evaluations. A Gaussian-process
surrogate has the form

$$
f(\theta) \sim \mathcal{GP}(\mu(\theta), k(\theta, \theta')).
$$

An acquisition function balances exploration and exploitation. Common choices
include expected improvement,

$$
EI(\theta) = \mathbb{E}[\max(0, f(\theta) - f^*)],
$$

and upper confidence bound,

$$
UCB(\theta) = \mu(\theta) + \kappa\sigma(\theta).
$$

The optimization loop is:

1. Fit the surrogate to observed pairs $\{(\theta_i, f(\theta_i))\}$.
2. Find the next $\theta$ that maximizes the acquisition function.
3. Evaluate $f(\theta)$ and add the observation to the data.
4. Repeat until the budget is exhausted.

Common tools include Optuna, Hyperopt with its Tree-structured Parzen Estimator,
Google Vizier, and Weights & Biases Sweeps.

**Advantages:** sample-efficient for expensive evaluations, provides principled
uncertainty estimates, and supports continuous, discrete, and categorical
variables.

**Limitations:** surrogate overhead grows with the number of observations,
smooth objective assumptions may not hold, and very high-dimensional spaces
above roughly 20 dimensions can be difficult.

### Multi-fidelity optimization

Multi-fidelity methods use a cheap approximation to guide search before
evaluating promising candidates at full fidelity. Fidelity can vary by training
time, data size, model size, or input resolution.

Successive halving trains all active configurations with a small budget, keeps
the best half, and doubles the budget until one remains:

```python
def successive_halving(configurations, budget):
    active_configs = configurations.copy()
    budget_per_config = budget // len(configurations)

    while len(active_configs) > 1:
        results = []
        for config in active_configs:
            score = train_model(config, budget=budget_per_config)
            results.append((config, score))

        results.sort(key=lambda item: item[1], reverse=True)
        active_configs = [config for config, _ in results[: len(results) // 2]]
        budget_per_config *= 2

    return active_configs[0]
```

ASHA extends successive halving to asynchronous parallel execution and is used
by many modern tuning frameworks.

### Population-based training

Population-based training (PBT), introduced by DeepMind in 2017, trains several
models with different hyperparameters simultaneously:

1. Initialize a population with different hyperparameters.
2. Train each model independently for a period.
3. Evaluate the population.
4. Replace weak models with copies of stronger models.
5. Perturb the copied models' hyperparameters.
6. Continue training and repeat.

PBT adapts hyperparameters online and can discover time-varying values. It is
particularly useful for long training runs, large language models,
reinforcement learning, and neural architecture search.

### Automated machine learning

Neural architecture search automates architecture discovery. Its search method
may use a reinforcement-learning controller, evolutionary mutation and
selection, or differentiable optimization over an architecture space.
Progressive search increases complexity gradually; weight sharing amortizes
training across candidates; one-shot methods train a supernet containing all
candidate architectures.

## Multi-objective and analysis methods

### Pareto optimization

A configuration is Pareto optimal when no other configuration improves one
objective without worsening another. The following routine marks the
non-dominated rows in a score matrix:

```python
import numpy as np


def is_pareto_optimal(scores, maximize=(True, False)):
    """Return a mask identifying non-dominated configurations."""
    is_efficient = np.ones(scores.shape[0], dtype=bool)

    for i, score in enumerate(scores):
        if is_efficient[i]:
            for j, other_score in enumerate(scores):
                if i != j and is_efficient[j]:
                    dominates = True
                    for k, maximize_k in enumerate(maximize):
                        if maximize_k and score[k] < other_score[k]:
                            dominates = False
                            break
                        if not maximize_k and score[k] > other_score[k]:
                            dominates = False
                            break
                    if dominates:
                        is_efficient[j] = False

    return is_efficient
```

### Hyperparameter importance

A random forest can estimate practical importance from completed trials:

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def analyze_hyperparameter_importance(X_params, y_scores):
    """Fit a model and report feature importance for search parameters."""
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_params, y_scores)

    importances = rf.feature_importances_
    param_names = ["learning_rate", "batch_size", "hidden_units", ...]
    sorted_idx = np.argsort(importances)[::-1]

    print("Hyperparameter Importance Ranking:")
    for i, idx in enumerate(sorted_idx):
        print(f"{i + 1}. {param_names[idx]}: {importances[idx]:.3f}")

    return importances
```

## Implementations

The generated [tuning API reference](api/tuning.md) documents the repository's
random search, Bayesian optimization, learning-rate finder, multi-fidelity,
population-based training, and integrated tuning framework implementations. The
framework combines optimization strategies with experiment tracking and
statistical analysis.
Learning-rate scheduler implementations are documented in the generated
[optimizer API reference](api/optimizers.md).
