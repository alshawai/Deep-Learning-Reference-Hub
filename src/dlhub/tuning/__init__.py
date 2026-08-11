"""
Hyperparameter Tuning
=====================

Search strategies for choosing hyperparameters, from random sampling through
model-based search to the multi-fidelity methods that spend their budget
unevenly on purpose.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

from dlhub.tuning.bayesian import (
    BayesianOptimizationResult,
    BayesianOptimizer,
    GaussianProcess,
    bayesian_optimize,
)

# One entry point per method -- `bayesian_optimize`, `random_search`,
# `asha_optimize`, `pbt_optimize` -- plus `optimize_hyperparameters`, which takes
# a method name and dispatches to one of them. The per-method functions take that
# method's own search-space format and return its own result type; the dispatcher
# takes the framework's and returns an `ExperimentResult`.
from dlhub.tuning.framework import (
    ExperimentConfig,
    ExperimentLogger,
    ExperimentResult,
    FunctionObjective,
    HyperparameterConfig,
    HyperparameterOptimizer,
    HyperparameterSampler,
    ObjectiveFunction,
    OptimizationMethod,
    TrialResult,
    optimize_hyperparameters,
)
from dlhub.tuning.learning_rate_finder import (
    BaseTrainer,
    FunctionTrainer,
    LearningRateFinder,
    LearningRateFinderResult,
    find_learning_rate,
    suggest_learning_rate_schedule,
)
from dlhub.tuning.multifidelity import (
    ASHAOptimizer,
    CandidateResult,
    FidelityConfig,
    FidelityEvaluator,
    FunctionEvaluator,
    MultiFidelityResult,
    analyze_fidelity_correlation,
    asha_optimize,
)
from dlhub.tuning.population_based import (
    ChoicePerturbation,
    FunctionWorker,
    HyperparameterDistribution,
    LogUniformPerturbation,
    PBTResult,
    PopulationBasedTrainer,
    UniformPerturbation,
    WorkerInterface,
    WorkerState,
    pbt_optimize,
)

# `random_search` names both a module here and a function inside it. Re-exporting
# the function would rebind the attribute and make `dlhub.tuning.random_search`
# resolve to it, so callers reach that one through the module path instead.
from dlhub.tuning.random_search import (
    ChoiceDistribution,
    IntegerDistribution,
    LogUniformDistribution,
    ParameterDistribution,
    PowerDistribution,
    RandomSearchOptimizer,
    RandomSearchResult,
    UniformDistribution,
    analyze_parameter_importance,
)

__all__ = [
    "ASHAOptimizer",
    "BaseTrainer",
    "BayesianOptimizationResult",
    "BayesianOptimizer",
    "CandidateResult",
    "ChoiceDistribution",
    "ChoicePerturbation",
    "ExperimentConfig",
    "ExperimentLogger",
    "ExperimentResult",
    "FidelityConfig",
    "FidelityEvaluator",
    "FunctionEvaluator",
    "FunctionObjective",
    "FunctionTrainer",
    "FunctionWorker",
    "GaussianProcess",
    "HyperparameterConfig",
    "HyperparameterDistribution",
    "HyperparameterOptimizer",
    "HyperparameterSampler",
    "IntegerDistribution",
    "LearningRateFinder",
    "LearningRateFinderResult",
    "LogUniformDistribution",
    "LogUniformPerturbation",
    "MultiFidelityResult",
    "ObjectiveFunction",
    "OptimizationMethod",
    "PBTResult",
    "ParameterDistribution",
    "PopulationBasedTrainer",
    "PowerDistribution",
    "RandomSearchOptimizer",
    "RandomSearchResult",
    "TrialResult",
    "UniformDistribution",
    "UniformPerturbation",
    "WorkerInterface",
    "WorkerState",
    "analyze_fidelity_correlation",
    "analyze_parameter_importance",
    "asha_optimize",
    "bayesian_optimize",
    "find_learning_rate",
    "optimize_hyperparameters",
    "pbt_optimize",
    "suggest_learning_rate_schedule",
]
