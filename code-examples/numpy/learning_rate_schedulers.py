"""
Learning Rate Scheduler Implementation
=====================================

A comprehensive implementation of various learning rate scheduling strategies
commonly used in deep learning optimization.

Learning rate scheduling is crucial for training stability and convergence.
This module provides multiple scheduling strategies with configurable parameters.

References
----------
- Loshchilov, I., & Hutter, F. (2016). SGDR: Stochastic Gradient Descent with Warm Restarts
- Smith, L. N. (2017). Cyclical Learning Rates for Training Neural Networks
- Goyal, P. et al. (2017). Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import numpy as np
import math
from typing import Dict, Optional, Any
from enum import Enum
import warnings


class SchedulerType(Enum):
    """Enumeration of different scheduling strategies."""

    CONSTANT = "constant"
    STEP_DECAY = "step_decay"
    EXPONENTIAL_DECAY = "exponential_decay"
    POLYNOMIAL_DECAY = "polynomial_decay"
    COSINE_ANNEALING = "cosine_annealing"
    COSINE_ANNEALING_WARM_RESTARTS = "cosine_annealing_warm_restarts"
    CYCLICAL = "cyclical"
    ONE_CYCLE = "one_cycle"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    WARMUP_COSINE = "warmup_cosine"
    LINEAR_WARMUP = "linear_warmup"
    CUSTOM = "custom"


class LearningRateScheduler:
    """
    Comprehensive Learning Rate Scheduler with multiple scheduling strategies.

    This class provides various learning rate scheduling strategies commonly used
    in deep learning training, including step decay, cosine annealing, cyclical
    learning rates, and warm restarts.

    Parameters
    ----------
    initial_lr : float
        Initial learning rate
    scheduler_type : SchedulerType
        Type of scheduling strategy to use
    total_steps : int, optional
        Total number of training steps (required for some schedulers)
    **kwargs
        Additional parameters specific to each scheduler type

    Attributes
    ----------
    current_lr : float
        Current learning rate
    step_count : int
        Number of steps taken
    history : list
        History of learning rates
    """

    def __init__(
        self,
        initial_lr: float,
        scheduler_type: SchedulerType = SchedulerType.CONSTANT,
        total_steps: Optional[int] = None,
        **kwargs,
    ):
        if initial_lr <= 0:
            raise ValueError(
                f"Initial learning rate must be positive, got {initial_lr}"
            )

        self.initial_lr = initial_lr
        self.scheduler_type = scheduler_type
        self.total_steps = total_steps
        self.kwargs = kwargs

        self.current_lr = initial_lr
        self.step_count = 0
        self.history = [initial_lr]

        self._plateau_count = 0
        self._best_metric = None
        self._cycle_count = 0
        self._restart_count = 0

        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate scheduler-specific parameters."""
        if self.scheduler_type == SchedulerType.STEP_DECAY:
            if "step_size" not in self.kwargs:
                raise ValueError("step_size required for STEP_DECAY scheduler")
            if "gamma" not in self.kwargs:
                self.kwargs["gamma"] = 0.1

        elif self.scheduler_type == SchedulerType.EXPONENTIAL_DECAY:
            if "gamma" not in self.kwargs:
                raise ValueError("gamma required for EXPONENTIAL_DECAY scheduler")

        elif self.scheduler_type == SchedulerType.POLYNOMIAL_DECAY:
            if "power" not in self.kwargs:
                self.kwargs["power"] = 1.0
            if self.total_steps is None:
                raise ValueError("total_steps required for POLYNOMIAL_DECAY scheduler")

        elif self.scheduler_type == SchedulerType.COSINE_ANNEALING:
            if "T_max" not in self.kwargs and self.total_steps is None:
                raise ValueError(
                    "Either T_max or total_steps required for COSINE_ANNEALING"
                )
            if "eta_min" not in self.kwargs:
                self.kwargs["eta_min"] = 0.0

        elif self.scheduler_type == SchedulerType.COSINE_ANNEALING_WARM_RESTARTS:
            if "T_0" not in self.kwargs:
                self.kwargs["T_0"] = 10
            if "T_mult" not in self.kwargs:
                self.kwargs["T_mult"] = 2
            if "eta_min" not in self.kwargs:
                self.kwargs["eta_min"] = 0.0

        elif self.scheduler_type == SchedulerType.CYCLICAL:
            if "base_lr" not in self.kwargs:
                self.kwargs["base_lr"] = self.initial_lr * 0.1
            if "max_lr" not in self.kwargs:
                self.kwargs["max_lr"] = self.initial_lr
            if "step_size_up" not in self.kwargs:
                self.kwargs["step_size_up"] = 2000
            if "mode" not in self.kwargs:
                self.kwargs["mode"] = "triangular"

        elif self.scheduler_type == SchedulerType.ONE_CYCLE:
            if "max_lr" not in self.kwargs:
                self.kwargs["max_lr"] = self.initial_lr * 10
            if self.total_steps is None:
                raise ValueError("total_steps required for ONE_CYCLE scheduler")
            if "pct_start" not in self.kwargs:
                self.kwargs["pct_start"] = 0.3
            if "anneal_strategy" not in self.kwargs:
                self.kwargs["anneal_strategy"] = "cos"

        elif self.scheduler_type == SchedulerType.REDUCE_ON_PLATEAU:
            if "factor" not in self.kwargs:
                self.kwargs["factor"] = 0.1
            if "patience" not in self.kwargs:
                self.kwargs["patience"] = 10
            if "threshold" not in self.kwargs:
                self.kwargs["threshold"] = 1e-4
            if "cooldown" not in self.kwargs:
                self.kwargs["cooldown"] = 0
            if "min_lr" not in self.kwargs:
                self.kwargs["min_lr"] = 0.0

        elif self.scheduler_type == SchedulerType.WARMUP_COSINE:
            if "warmup_steps" not in self.kwargs:
                self.kwargs["warmup_steps"] = 1000
            if self.total_steps is None:
                raise ValueError("total_steps required for WARMUP_COSINE scheduler")

        elif self.scheduler_type == SchedulerType.LINEAR_WARMUP:
            if "warmup_steps" not in self.kwargs:
                raise ValueError("warmup_steps required for LINEAR_WARMUP scheduler")

        elif self.scheduler_type == SchedulerType.CUSTOM:
            if "custom_func" not in self.kwargs:
                raise ValueError("custom_func required for CUSTOM scheduler")

    def step(self, metric: Optional[float] = None) -> float:
        """
        Update the learning rate for one step.

        Parameters
        ----------
        metric : float, optional
            Current metric value (required for REDUCE_ON_PLATEAU)

        Returns
        -------
        float
            Updated learning rate
        """
        self.step_count += 1

        if self.scheduler_type == SchedulerType.CONSTANT:
            self.current_lr = self.initial_lr

        elif self.scheduler_type == SchedulerType.STEP_DECAY:
            self.current_lr = self._step_decay()

        elif self.scheduler_type == SchedulerType.EXPONENTIAL_DECAY:
            self.current_lr = self._exponential_decay()

        elif self.scheduler_type == SchedulerType.POLYNOMIAL_DECAY:
            self.current_lr = self._polynomial_decay()

        elif self.scheduler_type == SchedulerType.COSINE_ANNEALING:
            self.current_lr = self._cosine_annealing()

        elif self.scheduler_type == SchedulerType.COSINE_ANNEALING_WARM_RESTARTS:
            self.current_lr = self._cosine_annealing_warm_restarts()

        elif self.scheduler_type == SchedulerType.CYCLICAL:
            self.current_lr = self._cyclical()

        elif self.scheduler_type == SchedulerType.ONE_CYCLE:
            self.current_lr = self._one_cycle()

        elif self.scheduler_type == SchedulerType.REDUCE_ON_PLATEAU:
            self.current_lr = self._reduce_on_plateau(metric)

        elif self.scheduler_type == SchedulerType.WARMUP_COSINE:
            self.current_lr = self._warmup_cosine()

        elif self.scheduler_type == SchedulerType.LINEAR_WARMUP:
            self.current_lr = self._linear_warmup()

        elif self.scheduler_type == SchedulerType.CUSTOM:
            self.current_lr = self._custom()

        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

        self.current_lr = max(self.current_lr, 0.0)
        self.history.append(self.current_lr)
        return self.current_lr

    def _step_decay(self) -> float:
        """Step decay scheduler."""
        step_size = self.kwargs["step_size"]
        gamma = self.kwargs["gamma"]
        return self.initial_lr * (gamma ** (self.step_count // step_size))

    def _exponential_decay(self) -> float:
        """Exponential decay scheduler."""
        gamma = self.kwargs["gamma"]
        return self.initial_lr * (gamma**self.step_count)

    def _polynomial_decay(self) -> float:
        """Polynomial decay scheduler."""
        power = self.kwargs["power"]
        if self.step_count >= self.total_steps:
            return 0.0
        return self.initial_lr * ((1 - self.step_count / self.total_steps) ** power)

    def _cosine_annealing(self) -> float:
        """Cosine annealing scheduler."""
        T_max = self.kwargs.get("T_max", self.total_steps)
        eta_min = self.kwargs["eta_min"]

        if T_max is None:
            T_max = self.total_steps

        return (
            eta_min
            + (self.initial_lr - eta_min)
            * (1 + math.cos(math.pi * self.step_count / T_max))
            / 2
        )

    def _cosine_annealing_warm_restarts(self) -> float:
        """Cosine annealing with warm restarts (SGDR)."""
        T_0 = self.kwargs["T_0"]
        T_mult = self.kwargs["T_mult"]
        eta_min = self.kwargs["eta_min"]

        # Find which cycle we're in
        T_cur = self.step_count
        T_i = T_0

        while T_cur >= T_i:
            T_cur -= T_i
            T_i *= T_mult
            self._restart_count += 1

        return (
            eta_min
            + (self.initial_lr - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
        )

    def _cyclical(self) -> float:
        """Cyclical learning rate scheduler."""
        base_lr = self.kwargs["base_lr"]
        max_lr = self.kwargs["max_lr"]
        step_size_up = self.kwargs["step_size_up"]
        mode = self.kwargs["mode"]

        cycle = math.floor(1 + self.step_count / (2 * step_size_up))
        x = abs(self.step_count / step_size_up - 2 * cycle + 1)

        if mode == "triangular":
            scale_fn = lambda x: 1.0
            scale_mode = "cycle"
        elif mode == "triangular2":
            scale_fn = lambda x: 1 / (2.0 ** (cycle - 1))
            scale_mode = "cycle"
        elif mode == "exp_range":
            gamma = self.kwargs.get("gamma", 1.0)
            scale_fn = lambda x: gamma**self.step_count
            scale_mode = "iterations"
        else:
            raise ValueError(f"Unknown cyclical mode: {mode}")

        if scale_mode == "cycle":
            scale_factor = scale_fn(cycle)
        else:
            scale_factor = scale_fn(self.step_count)

        return base_lr + (max_lr - base_lr) * max(0, (1 - x)) * scale_factor

    def _one_cycle(self) -> float:
        """One cycle learning rate scheduler."""
        max_lr = self.kwargs["max_lr"]
        pct_start = self.kwargs["pct_start"]
        anneal_strategy = self.kwargs["anneal_strategy"]

        step_ratio = self.step_count / self.total_steps

        if step_ratio <= pct_start:
            # Warmup phase
            if anneal_strategy == "linear":
                return (
                    self.initial_lr
                    + (max_lr - self.initial_lr) * step_ratio / pct_start
                )
            else:  # cosine
                return (
                    self.initial_lr
                    + (max_lr - self.initial_lr)
                    * (1 - math.cos(math.pi * step_ratio / pct_start))
                    / 2
                )
        else:
            # Annealing phase
            remaining_ratio = (step_ratio - pct_start) / (1 - pct_start)
            if anneal_strategy == "linear":
                return max_lr - (max_lr - self.initial_lr) * remaining_ratio
            else:  # cosine
                return (
                    self.initial_lr
                    + (max_lr - self.initial_lr)
                    * (1 + math.cos(math.pi * remaining_ratio))
                    / 2
                )

    def _reduce_on_plateau(self, metric: Optional[float]) -> float:
        """Reduce on plateau scheduler."""
        if metric is None:
            warnings.warn("Metric required for REDUCE_ON_PLATEAU scheduler")
            return self.current_lr

        factor = self.kwargs["factor"]
        patience = self.kwargs["patience"]
        threshold = self.kwargs["threshold"]
        cooldown = self.kwargs["cooldown"]
        min_lr = self.kwargs["min_lr"]
        mode = self.kwargs.get("mode", "min")

        if self._best_metric is None:
            self._best_metric = metric
            return self.current_lr

        # Check if metric improved
        if mode == "min":
            improved = metric < self._best_metric - threshold
        else:  # mode == 'max'
            improved = metric > self._best_metric + threshold

        if improved:
            self._best_metric = metric
            self._plateau_count = 0
        else:
            self._plateau_count += 1

        # Reduce learning rate if patience exceeded
        if self._plateau_count > patience:
            new_lr = max(self.current_lr * factor, min_lr)
            if new_lr < self.current_lr:
                self._plateau_count = 0
            return new_lr

        return self.current_lr

    def _warmup_cosine(self) -> float:
        """Warmup followed by cosine annealing."""
        warmup_steps = self.kwargs["warmup_steps"]

        if self.step_count <= warmup_steps:
            # Linear warmup
            return self.initial_lr * self.step_count / warmup_steps
        else:
            # Cosine annealing
            progress = (self.step_count - warmup_steps) / (
                self.total_steps - warmup_steps
            )
            return self.initial_lr * (1 + math.cos(math.pi * progress)) / 2

    def _linear_warmup(self) -> float:
        """Linear warmup scheduler."""
        warmup_steps = self.kwargs["warmup_steps"]

        if self.step_count <= warmup_steps:
            return self.initial_lr * self.step_count / warmup_steps
        else:
            return self.initial_lr

    def _custom(self) -> float:
        """Create custom scheduler using user-provided function."""
        custom_func = self.kwargs["custom_func"]
        return custom_func(self.step_count, self.initial_lr, **self.kwargs)

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr

    def reset(self) -> None:
        """Reset scheduler to initial state."""
        self.current_lr = self.initial_lr
        self.step_count = 0
        self.history = [self.initial_lr]
        self._plateau_count = 0
        self._best_metric = None
        self._cycle_count = 0
        self._restart_count = 0

    def get_config(self) -> Dict[str, Any]:
        """Get scheduler configuration."""
        return {
            "initial_lr": self.initial_lr,
            "scheduler_type": self.scheduler_type.value,
            "total_steps": self.total_steps,
            "step_count": self.step_count,
            "kwargs": self.kwargs.copy(),
        }

    def get_state(self) -> Dict[str, Any]:
        """Get complete scheduler state."""
        return {
            "config": self.get_config(),
            "current_lr": self.current_lr,
            "history": self.history.copy(),
            "plateau_count": self._plateau_count,
            "best_metric": self._best_metric,
            "cycle_count": self._cycle_count,
            "restart_count": self._restart_count,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load scheduler state."""
        config = state["config"]
        self.initial_lr = config["initial_lr"]
        self.scheduler_type = SchedulerType(config["scheduler_type"])
        self.total_steps = config["total_steps"]
        self.step_count = config["step_count"]
        self.kwargs = config["kwargs"]

        self.current_lr = state["current_lr"]
        self.history = state["history"]
        self._plateau_count = state["plateau_count"]
        self._best_metric = state["best_metric"]
        self._cycle_count = state["cycle_count"]
        self._restart_count = state["restart_count"]


# Factory functions for common schedulers
def create_step_scheduler(
    initial_lr: float, step_size: int, gamma: float = 0.1
) -> LearningRateScheduler:
    """Create step decay scheduler."""
    return LearningRateScheduler(
        initial_lr=initial_lr,
        scheduler_type=SchedulerType.STEP_DECAY,
        step_size=step_size,
        gamma=gamma,
    )


def create_cosine_scheduler(
    initial_lr: float, total_steps: int, eta_min: float = 0.0
) -> LearningRateScheduler:
    """Create cosine annealing scheduler."""
    return LearningRateScheduler(
        initial_lr=initial_lr,
        scheduler_type=SchedulerType.COSINE_ANNEALING,
        total_steps=total_steps,
        eta_min=eta_min,
    )


def create_one_cycle_scheduler(
    initial_lr: float, max_lr: float, total_steps: int, pct_start: float = 0.3
) -> LearningRateScheduler:
    """Create one cycle scheduler."""
    return LearningRateScheduler(
        initial_lr=initial_lr,
        scheduler_type=SchedulerType.ONE_CYCLE,
        total_steps=total_steps,
        max_lr=max_lr,
        pct_start=pct_start,
    )


def create_warmup_cosine_scheduler(
    initial_lr: float, total_steps: int, warmup_steps: int
) -> LearningRateScheduler:
    """Create warmup + cosine scheduler."""
    return LearningRateScheduler(
        initial_lr=initial_lr,
        scheduler_type=SchedulerType.WARMUP_COSINE,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )


if __name__ == "__main__":
    print("Learning Rate Scheduler Test")
    print("=" * 50)

    print("\nTest 1: Step Decay Scheduler")
    step_scheduler = create_step_scheduler(initial_lr=0.1, step_size=10, gamma=0.5)

    print("Step\tLearning Rate")
    for step in range(25):
        lr = step_scheduler.step()
        if step % 5 == 0:
            print(f"{step}\t{lr:.6f}")

    print("\nTest 2: Cosine Annealing Scheduler")
    cosine_scheduler = create_cosine_scheduler(
        initial_lr=0.1, total_steps=50, eta_min=0.001
    )

    cosine_lrs = []
    for step in range(50):
        cosine_lrs.append(cosine_scheduler.step())

    print(f"Initial LR: {cosine_lrs[0]:.6f}")
    print(f"Middle LR: {cosine_lrs[25]:.6f}")
    print(f"Final LR: {cosine_lrs[-1]:.6f}")

    print("\nTest 3: One Cycle Scheduler")
    one_cycle_scheduler = create_one_cycle_scheduler(
        initial_lr=0.01, max_lr=0.1, total_steps=100, pct_start=0.3
    )

    one_cycle_lrs = []
    for step in range(100):
        one_cycle_lrs.append(one_cycle_scheduler.step())

    max_lr_idx = np.argmax(one_cycle_lrs)
    print(f"Max LR: {max(one_cycle_lrs):.6f} at step {max_lr_idx}")
    print(f"Final LR: {one_cycle_lrs[-1]:.6f}")

    print("\nTest 4: Reduce on Plateau Scheduler")
    plateau_scheduler = LearningRateScheduler(
        initial_lr=0.1,
        scheduler_type=SchedulerType.REDUCE_ON_PLATEAU,
        factor=0.5,
        patience=3,
        threshold=0.01,
    )

    # Simulate training with plateauing loss
    losses = [1.0, 0.9, 0.8, 0.79, 0.79, 0.79, 0.79, 0.4, 0.3]
    print("Step\tLoss\tLearning Rate")
    for step, loss in enumerate(losses):
        lr = plateau_scheduler.step(metric=loss)
        print(f"{step}\t{loss:.2f}\t{lr:.6f}")

    print("\nAll scheduler tests completed successfully!")

    # -> Feel free to add test to other schedulers as well!
