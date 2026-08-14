# The hyperparameter tuning landscape

Hyperparameter tuning remains one of the most critical and difficult parts of
deep learning. Deep learning has achieved tremendous success, but training a
model still requires a combination of informed judgment and systematic search.

Unlike model parameters such as weights and biases, which are learned during
training, hyperparameters configure the training process or model before it
begins. Their values can dramatically affect model performance, training
stability, and convergence speed.

## Why tuning matters

Different hyperparameters influence different parts of the learning problem:

- The learning rate affects convergence speed and final accuracy.
- Architecture choices determine model capacity and expressiveness.
- Regularization controls the bias-variance tradeoff.
- Batch size influences gradient estimates and memory requirements.

The consequences become more pronounced for large language models and
transformers:

- **Computational cost:** training large models is expensive, so inefficient
  trials consume substantial resources.
- **Emergent behavior:** deep networks contain complex interactions between
  hyperparameters.
- **Scale sensitivity:** values that work for small models may fail at larger
  scales.

This uneven influence is why tuning should be prioritized. Learning rate,
architecture, and batch size usually deserve attention before numerical
stability constants or small changes to decay factors. The complete priority
order and typical ranges are listed in the
[hyperparameter search methods reference](../reference/hyperparameter-search-methods.md#hyperparameters-by-impact).

## Why random search often beats grid search

Grid search is systematic, reproducible, and easy to parallelize, but it spends
the same number of trials along every dimension. Its cost therefore grows
exponentially with the number of hyperparameters, and its fixed points can miss
good values between grid coordinates.

Random search works better when only a few dimensions strongly affect the
result. This low *effective dimensionality* is common in deep-learning search
spaces: random sampling explores more distinct values of every important
dimension instead of repeatedly pairing the same values with changes to
unimportant dimensions. Bergstra and Bengio highlighted this advantage in
2012.

Random search is still not universally best. Bayesian optimization can be more
sample-efficient when evaluations are expensive, while multi-fidelity methods
can stop weak configurations before spending a full training budget. The
[method catalog](../reference/hyperparameter-search-methods.md#search-methods)
compares these alternatives.

## Search is a resource-allocation problem

The objective is not merely to find good hyperparameters. It is to find them
efficiently and reliably at the scale of the intended training run.
Multi-fidelity optimization makes this explicit by using cheaper approximations
such as shorter training time, less data, smaller models, or lower-resolution
inputs. Promising candidates receive more resources; weak candidates are
discarded.

Population-based training goes further by changing hyperparameters during
training. A population trains in parallel, weak members copy stronger members,
and copied hyperparameters are perturbed. This can discover schedules rather
than a single fixed configuration, which is particularly useful for long
language-model and reinforcement-learning runs.

AutoML broadens the search target to architecture itself. Neural architecture
search may use a controller trained by reinforcement learning, evolutionary
mutation and selection, or differentiable optimization over the architecture
space. Progressive search, weight sharing, and one-shot supernets reduce the
otherwise prohibitive cost.

## Objectives beyond accuracy

Real systems often optimize competing objectives:

- accuracy against inference latency;
- accuracy against model size and memory use;
- accuracy against fairness or bias metrics.

Multi-objective search therefore looks for a *Pareto frontier*: the set of
non-dominated configurations for which no objective can improve without
worsening another. There may be no single best configuration until deployment
constraints select a point on that frontier.

## Reusing search knowledge

Search results need not be discarded after one project. Transfer learning for
hyperparameters uses earlier experiments to initialize later searches:

1. **Meta-learning** learns a mapping from dataset or architecture
   characteristics to promising hyperparameters.
2. **Warm-starting** initializes Bayesian optimization with results from similar
   problems.
3. **Few-shot hyperparameter optimization** adapts configurations quickly for a
   new task.

Hyperparameter importance analysis provides another form of reusable knowledge.
Functional ANOVA decomposes the objective into individual and interaction
effects:

$$
f(\theta) = f_0 + \sum_i f_i(\theta_i)
  + \sum_{i<j} f_{ij}(\theta_i, \theta_j) + \ldots
$$

The decomposition identifies which dimensions consumed search budget without
materially changing the outcome. Future searches can narrow or remove those
dimensions and concentrate on influential interactions.

## What reliable tuning requires

Effective tuning combines principled optimization with domain knowledge and
resource awareness. It also requires statistical rigor: comparing single runs
can select noise rather than a better configuration, while repeatedly consulting
one validation set can overfit the search process to that set.

The field and its tooling evolve quickly. Revisit defaults, scaling assumptions,
and method choices as architectures and available compute change; automation is
useful, but it does not remove the need to understand the optimization method or
the evidence behind a selected configuration.

Start with the [learning-rate tuning guide](../how-to/tune-learning-rate.md) when
the learning rate is the main uncertainty. For a complete experiment, follow
[how to run a hyperparameter search](../how-to/run-hyperparameter-search.md).
