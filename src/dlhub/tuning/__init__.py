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
    "FidelityConfig",
    "FidelityEvaluator",
    "FunctionEvaluator",
    "FunctionTrainer",
    "FunctionWorker",
    "GaussianProcess",
    "HyperparameterDistribution",
    "IntegerDistribution",
    "LearningRateFinder",
    "LearningRateFinderResult",
    "LogUniformDistribution",
    "LogUniformPerturbation",
    "MultiFidelityResult",
    "PBTResult",
    "ParameterDistribution",
    "PopulationBasedTrainer",
    "PowerDistribution",
    "RandomSearchOptimizer",
    "RandomSearchResult",
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
