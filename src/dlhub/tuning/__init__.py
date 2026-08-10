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
    optimize_hyperparameters,
)

# `framework.optimize_hyperparameters` is a second function of that name: it drives
# whichever strategy its `optimization_method` argument selects, over a different
# search-space format, and returns a different result type. Only the Bayesian one
# is re-exported here; this one is reached through its module path until Phase 2
# settles which of the two is canonical.
from dlhub.tuning.framework import (
    ExperimentConfig,
    ExperimentLogger,
    FunctionObjective,
    HyperparameterConfig,
    HyperparameterOptimizer,
    HyperparameterSampler,
    ObjectiveFunction,
    OptimizationMethod,
    OptimizationResult,
    TrialResult,
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
    "OptimizationResult",
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
    "find_learning_rate",
    "optimize_hyperparameters",
    "pbt_optimize",
    "suggest_learning_rate_schedule",
]
