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
    "ChoiceDistribution",
    "IntegerDistribution",
    "LogUniformDistribution",
    "ParameterDistribution",
    "PowerDistribution",
    "RandomSearchOptimizer",
    "RandomSearchResult",
    "UniformDistribution",
    "analyze_parameter_importance",
]
