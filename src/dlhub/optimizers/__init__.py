"""
Optimizers
==========

Gradient descent and the adaptive methods built on top of it.

Each optimizer is written to be read alongside the explanation that derives it,
so the update rule appears in the code in the same form it appears in the
mathematics.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

from dlhub.optimizers.adam import AdamOptimizer, create_adam_optimizer
from dlhub.optimizers.exponential_weighted_averages import (
    AveragingStrategy,
    ExponentialWeightedAverage,
    MultiVariateEWA,
    create_adam_ewa_pair,
    create_momentum_ewa,
    create_rmsprop_ewa,
)
from dlhub.optimizers.mini_batch import MiniBatchGradientDescent
from dlhub.optimizers.momentum import MomentumOptimizer
from dlhub.optimizers.rmsprop import RMSpropOptimizer
from dlhub.optimizers.schedules import (
    LearningRateScheduler,
    SchedulerType,
    create_cosine_scheduler,
    create_one_cycle_scheduler,
    create_step_scheduler,
    create_warmup_cosine_scheduler,
)

__all__ = [
    "AdamOptimizer",
    "AveragingStrategy",
    "ExponentialWeightedAverage",
    "LearningRateScheduler",
    "MiniBatchGradientDescent",
    "MomentumOptimizer",
    "MultiVariateEWA",
    "RMSpropOptimizer",
    "SchedulerType",
    "create_adam_ewa_pair",
    "create_adam_optimizer",
    "create_cosine_scheduler",
    "create_momentum_ewa",
    "create_one_cycle_scheduler",
    "create_rmsprop_ewa",
    "create_step_scheduler",
    "create_warmup_cosine_scheduler",
]
