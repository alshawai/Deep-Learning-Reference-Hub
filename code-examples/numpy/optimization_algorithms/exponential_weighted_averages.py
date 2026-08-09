"""
Exponential Weighted Averages (EWA) Implementation
=================================================

A comprehensive implementation of exponential weighted averages with bias correction,
multiple averaging strategies, and numerical stability features.

Exponential weighted averages are fundamental building blocks for modern optimization
algorithms like Adam, RMSprop, and momentum-based optimizers.

References
----------
- Used in Adam, RMSprop, Momentum optimizers
- Bias correction technique from Adam paper (Kingma & Ba, 2014)

Author
------
Deep Learning Reference Hub

License
-------
MIT

Notes
-----
Bias correction divides the accumulator by the total weight it has applied to its
samples. Textbooks write that weight as ``1 - beta**t``, which is its closed form
when beta is the same at every step.

This implementation tracks the weight directly, through
``weight = beta_t * weight + (1 - beta_t)``. The two agree exactly for a constant
beta. They diverge once beta varies with t, which is what ``warmup_steps`` and the
``EXPONENTIAL_DECAY`` strategy both do: there the closed form understates the
applied weight and the correction inflates the result.
"""

import warnings
from enum import Enum
from typing import Any

import numpy as np


class AveragingStrategy(Enum):
    """Enumeration of different averaging strategies."""

    SIMPLE = "simple"
    BIAS_CORRECTED = "bias_corrected"
    VARIANCE_CORRECTED = "variance_corrected"
    EXPONENTIAL_DECAY = "exponential_decay"


class ExponentialWeightedAverage:
    """
    Exponential Weighted Average with bias correction and multiple strategies.

    This class implements exponential weighted averages (also known as exponentially
    weighted moving averages) with various correction techniques commonly used in
    deep learning optimization algorithms.

    The basic formula is:
        v_t = beta * v_{t-1} + (1 - beta) * theta_t

    Where:
        - v_t is the average at time t
        - beta is the decay parameter
        - theta_t is the current value
        - v_0 = 0 (initial value)

    Parameters
    ----------
    beta : float, default=0.9
        Decay parameter (0 < beta < 1). Higher values give more weight to past values.
        Common values: 0.9 (momentum), 0.999 (second moments in Adam)
    bias_correction : bool, default=True
        Whether to apply bias correction to account for initialization bias
    strategy : AveragingStrategy, default=AveragingStrategy.BIAS_CORRECTED
        Averaging strategy to use
    epsilon : float, default=1e-8
        Small constant for numerical stability
    warmup_steps : int, default=0
        Number of warmup steps before applying full averaging

    Attributes
    ----------
    v : float or np.ndarray
        Current average value
    t : int
        Time step (number of updates)
    history : list
        History of average values
    """

    def __init__(
        self,
        beta: float = 0.9,
        bias_correction: bool = True,
        strategy: AveragingStrategy = AveragingStrategy.BIAS_CORRECTED,
        epsilon: float = 1e-8,
        warmup_steps: int = 0,
    ):
        if not 0.0 < beta < 1.0:
            raise ValueError(f"Beta must be in (0, 1), got {beta}")
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        if warmup_steps < 0:
            raise ValueError(f"Warmup steps must be non-negative, got {warmup_steps}")

        self.beta = beta
        self.bias_correction = bias_correction
        self.strategy = strategy
        self.epsilon = epsilon
        self.warmup_steps = warmup_steps

        self.v = None
        self.t = 0
        self.history = []

        # Cumulative weight the accumulator has actually applied to its samples.
        # The textbook correction factor 1 - beta**t is this quantity's closed form
        # for a *constant* beta. Warm-up varies beta with t, which invalidates that
        # closed form, so the weight is tracked directly instead. See Notes.
        self.weight = 0.0

        self.squared_avg = None
        self.variance_history = []

    def update(self, value: float | np.ndarray) -> float | np.ndarray:
        """
        Update the exponential weighted average with a new value.

        Parameters
        ----------
        value : float or np.ndarray
            New value to incorporate into the average

        Returns
        -------
        float or np.ndarray
            Updated average value
        """
        self.t += 1

        if isinstance(value, np.ndarray):
            if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                warnings.warn("NaN or Inf detected in input value")
                value = np.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6)
        else:
            if np.isnan(value) or np.isinf(value):
                warnings.warn("NaN or Inf detected in input value")
                value = 0.0 if np.isnan(value) else (1e6 if value > 0 else -1e6)

        if self.v is None:
            if isinstance(value, np.ndarray):
                self.v = np.zeros_like(value, dtype=np.float64)
                if self.strategy == AveragingStrategy.VARIANCE_CORRECTED:
                    self.squared_avg = np.zeros_like(value, dtype=np.float64)
            else:
                self.v = 0.0
                if self.strategy == AveragingStrategy.VARIANCE_CORRECTED:
                    self.squared_avg = 0.0

        if self.t <= self.warmup_steps:
            effective_beta = min(self.beta, (self.t - 1) / self.t)
        else:
            effective_beta = self.beta

        if self.strategy == AveragingStrategy.EXPONENTIAL_DECAY:
            effective_beta = effective_beta ** (self.t / 1000)  # Decay over time

        # Track the weight with whichever coefficient this step actually applied,
        # so the two stay consistent for every strategy.
        self.weight = effective_beta * self.weight + (1 - effective_beta)

        if self.strategy == AveragingStrategy.SIMPLE:
            self.v = effective_beta * self.v + (1 - effective_beta) * value
            result = self.v

        elif self.strategy == AveragingStrategy.BIAS_CORRECTED:
            self.v = effective_beta * self.v + (1 - effective_beta) * value
            if self.bias_correction:
                result = self.v / (self.weight + self.epsilon)
            else:
                result = self.v

        elif self.strategy == AveragingStrategy.VARIANCE_CORRECTED:
            self.v = effective_beta * self.v + (1 - effective_beta) * value
            self.squared_avg = effective_beta * self.squared_avg + (
                1 - effective_beta
            ) * (value**2)

            if self.bias_correction:
                mean_corrected = self.v / (self.weight + self.epsilon)
                variance_corrected = self.squared_avg / (self.weight + self.epsilon)

                variance = variance_corrected - mean_corrected**2
                self.variance_history.append(variance)
                result = mean_corrected
            else:
                result = self.v

        elif self.strategy == AveragingStrategy.EXPONENTIAL_DECAY:
            self.v = effective_beta * self.v + (1 - effective_beta) * value
            result = self.v

        else:
            raise ValueError(f"Unknown averaging strategy: {self.strategy}")

        self.history.append(result.copy() if isinstance(result, np.ndarray) else result)
        return result

    def get_current_average(self) -> float | np.ndarray | None:
        """
        Get the current average value.

        Returns
        -------
        float, np.ndarray, or None
            Current average value, None if no updates have been made
        """
        if self.v is None:
            return None

        if self.strategy == AveragingStrategy.BIAS_CORRECTED and self.bias_correction:
            return self.v / (self.weight + self.epsilon)
        else:
            return self.v

    def get_variance(self) -> float | np.ndarray | None:
        """
        Get the current variance estimate (only available with VARIANCE_CORRECTED strategy).

        Returns
        -------
        float, np.ndarray, or None
            Current variance estimate, None if not available
        """
        if (
            self.strategy != AveragingStrategy.VARIANCE_CORRECTED
            or self.squared_avg is None
        ):
            return None

        if self.bias_correction:
            mean_corrected = self.v / (self.weight + self.epsilon)
            variance_corrected = self.squared_avg / (self.weight + self.epsilon)
            return variance_corrected - mean_corrected**2
        else:
            return self.squared_avg - self.v**2

    def reset(self) -> None:
        """Reset the exponential weighted average to initial state."""
        self.v = None
        self.squared_avg = None
        self.t = 0
        self.weight = 0.0
        self.history.clear()
        self.variance_history.clear()

    def get_effective_window_size(self) -> float:
        """
        Get the effective window size of the exponential weighted average.

        The effective window size is approximately 1/(1-beta).

        Returns
        -------
        float
            Effective window size
        """
        return 1.0 / (1.0 - self.beta)

    def get_config(self) -> dict[str, Any]:
        """
        Get configuration dictionary.

        Returns
        -------
        dict
            Configuration dictionary
        """
        return {
            "beta": self.beta,
            "bias_correction": self.bias_correction,
            "strategy": self.strategy.value,
            "epsilon": self.epsilon,
            "warmup_steps": self.warmup_steps,
            "time_step": self.t,
        }

    def get_state(self) -> dict[str, Any]:
        """
        Get complete state dictionary.

        Returns
        -------
        dict
            Complete state dictionary
        """
        return {
            "config": self.get_config(),
            "v": self.v,
            "weight": self.weight,
            "squared_avg": self.squared_avg,
            "history": self.history.copy(),
            "variance_history": self.variance_history.copy(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """
        Load state from dictionary.

        Parameters
        ----------
        state : dict
            State dictionary from get_state()
        """
        config = state["config"]
        self.beta = config["beta"]
        self.bias_correction = config["bias_correction"]
        self.strategy = AveragingStrategy(config["strategy"])
        self.epsilon = config["epsilon"]
        self.warmup_steps = config["warmup_steps"]
        self.t = config["time_step"]

        self.v = state["v"]
        # Older states predate the tracked weight. Fall back to the constant-beta
        # closed form, which is exact whenever warmup_steps is 0.
        self.weight = state.get("weight", 1 - self.beta**self.t if self.t else 0.0)
        self.squared_avg = state["squared_avg"]
        self.history = state["history"].copy()
        self.variance_history = state["variance_history"].copy()


class MultiVariateEWA:
    """
    Multi-variate Exponential Weighted Average for handling multiple variables simultaneously.

    This class manages multiple exponential weighted averages, commonly used in
    optimization algorithms where different parameters need separate averages.

    Parameters
    ----------
    beta : float, default=0.9
        Common decay parameter for all variables
    bias_correction : bool, default=True
        Whether to apply bias correction
    strategy : AveragingStrategy, default=AveragingStrategy.BIAS_CORRECTED
        Averaging strategy
    **kwargs
        Additional parameters passed to individual EWA instances
    """

    def __init__(
        self,
        beta: float = 0.9,
        bias_correction: bool = True,
        strategy: AveragingStrategy = AveragingStrategy.BIAS_CORRECTED,
        **kwargs,
    ):
        self.beta = beta
        self.bias_correction = bias_correction
        self.strategy = strategy
        self.kwargs = kwargs

        self.averages: dict[str, ExponentialWeightedAverage] = {}

    def update(
        self, values: dict[str, float | np.ndarray]
    ) -> dict[str, float | np.ndarray]:
        """
        Update all averages with new values.

        Parameters
        ----------
        values : dict
            Dictionary of new values for each variable

        Returns
        -------
        dict
            Dictionary of updated averages
        """
        results = {}

        for name, value in values.items():
            if name not in self.averages:
                self.averages[name] = ExponentialWeightedAverage(
                    beta=self.beta,
                    bias_correction=self.bias_correction,
                    strategy=self.strategy,
                    **self.kwargs,
                )

            results[name] = self.averages[name].update(value)

        return results

    def get_averages(self) -> dict[str, float | np.ndarray]:
        """Get current averages for all variables."""
        return {name: ewa.get_current_average() for name, ewa in self.averages.items()}

    def reset(self) -> None:
        """Reset all averages."""
        for ewa in self.averages.values():
            ewa.reset()

    def get_state(self) -> dict[str, Any]:
        """Get complete state for all averages."""
        return {
            "config": {
                "beta": self.beta,
                "bias_correction": self.bias_correction,
                "strategy": self.strategy.value,
                "kwargs": self.kwargs,
            },
            "averages": {name: ewa.get_state() for name, ewa in self.averages.items()},
        }


def create_momentum_ewa(beta: float = 0.9) -> ExponentialWeightedAverage:
    """Create EWA for momentum optimization."""
    return ExponentialWeightedAverage(
        beta=beta, bias_correction=True, strategy=AveragingStrategy.BIAS_CORRECTED
    )


def create_rmsprop_ewa(beta: float = 0.999) -> ExponentialWeightedAverage:
    """Create EWA for RMSprop (second moments)."""
    return ExponentialWeightedAverage(
        beta=beta, bias_correction=True, strategy=AveragingStrategy.BIAS_CORRECTED
    )


def create_adam_ewa_pair(
    beta1: float = 0.9, beta2: float = 0.999
) -> tuple[ExponentialWeightedAverage, ExponentialWeightedAverage]:
    """Create EWA pair for Adam optimizer (first and second moments)."""
    first_moment = ExponentialWeightedAverage(
        beta=beta1, bias_correction=True, strategy=AveragingStrategy.BIAS_CORRECTED
    )

    second_moment = ExponentialWeightedAverage(
        beta=beta2, bias_correction=True, strategy=AveragingStrategy.BIAS_CORRECTED
    )

    return first_moment, second_moment


if __name__ == "__main__":
    print("Exponential Weighted Averages Test")
    print("=" * 50)
    print("\nTest 1: Simple EWA with noisy signal")
    np.random.seed(42)

    true_signal = np.sin(np.linspace(0, 4 * np.pi, 100))
    noisy_signal = true_signal + 0.3 * np.random.randn(100)

    betas = [0.5, 0.9, 0.99]
    results = {}

    for beta in betas:
        ewa = ExponentialWeightedAverage(beta=beta, bias_correction=True)
        smoothed = []

        for value in noisy_signal:
            smoothed.append(ewa.update(value))

        results[f"beta_{beta}"] = smoothed
        print(
            f"  Beta {beta}: Effective window = {ewa.get_effective_window_size():.1f}"
        )

    print("\nTest 2: Multi-variate EWA")
    multi_ewa = MultiVariateEWA(beta=0.9, bias_correction=True)

    for step in range(10):
        values = {
            "W1": np.random.randn(2, 3) * 0.1,
            "b1": np.random.randn(2, 1) * 0.01,
            "W2": np.random.randn(1, 2) * 0.1,
        }
        multi_ewa.update(values)
        averages = multi_ewa.get_averages()

        if step % 3 == 0:
            print(f"  Step {step}: {averages}")

    print("\nTest 3: Variance-corrected EWA")
    var_ewa = ExponentialWeightedAverage(
        beta=0.9, strategy=AveragingStrategy.VARIANCE_CORRECTED, bias_correction=True
    )

    for i in range(20):
        value = np.random.randn() * (1 + 0.1 * i)  # Increasing variance
        avg = var_ewa.update(value)
        var = var_ewa.get_variance()

        if i % 5 == 0:
            print(f"  Step {i}: Average = {avg:.4f}, Variance = {var:.4f}")

    print(f"Final configuration: {var_ewa.get_config()}")
    print("\nAll EWA tests completed successfully!")
