"""
Learning Rate Finder for Hyperparameter Tuning
===============================================

Automated learning rate range testing to find optimal learning rate
ranges before full training. This technique helps identify good learning
rate ranges by monitoring loss behavior during short training runs with
exponentially increasing learning rates.

References
----------
- Smith, L. N. (2017). "Cyclical Learning Rates for Training Neural Networks."
  IEEE Winter Conference on Applications of Computer Vision (WACV).
- Smith, L. N. (2018). "A disciplined approach to neural network hyper-parameters:
  Part 1 -- learning rate, batch size, momentum, and weight decay." arXiv preprint.

Author
------
Deep Learning Reference Hub

License
-------
MIT License

Notes
-----
The learning rate finder is particularly useful for:
1. Finding the maximum usable learning rate
2. Identifying learning rate ranges for cyclical learning rate schedules
3. Detecting when the learning rate is too high (loss divergence)
4. Setting appropriate learning rates for different optimizers
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Callable, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class LearningRateFinderResult:
    """
    Container for learning rate finder results.

    Attributes
    ----------
    learning_rates : np.ndarray
        Array of learning rates tested
    losses : np.ndarray
        Corresponding loss values
    smoothed_losses : np.ndarray
        Smoothed loss values for trend analysis
    suggested_lr : float
        Suggested learning rate based on analysis
    min_gradient_lr : float
        Learning rate with steepest loss decrease
    analysis : dict
        Additional analysis metrics and diagnostics
    """

    learning_rates: np.ndarray
    losses: np.ndarray
    smoothed_losses: np.ndarray
    suggested_lr: float
    min_gradient_lr: float
    analysis: Dict[str, Any]


class BaseTrainer(ABC):
    """
    Abstract base class for training interface.

    Defines the interface that training functions must implement
    to work with the learning rate finder.
    """

    @abstractmethod
    def train_batch(self, learning_rate: float) -> float:
        """
        Train one batch with given learning rate and return loss.

        Parameters
        ----------
        learning_rate : float
            Learning rate to use for this batch

        Returns
        -------
        float
            Loss value after training step
        """
        pass

    @abstractmethod
    def reset_model(self) -> None:
        """Reset model to initial state."""
        pass


class FunctionTrainer(BaseTrainer):
    """
    Trainer wrapper for function-based training.

    Wraps user-provided training and reset functions to conform
    to the BaseTrainer interface.

    Parameters
    ----------
    train_function : callable
        Function that takes learning_rate and returns loss
    reset_function : callable
        Function to reset model state
    """

    def __init__(
        self,
        train_function: Callable[[float], float],
        reset_function: Callable[[], None],
    ):
        self.train_function = train_function
        self.reset_function = reset_function

    def train_batch(self, learning_rate: float) -> float:
        """Train one batch with given learning rate."""
        return self.train_function(learning_rate)

    def reset_model(self) -> None:
        """Reset model to initial state."""
        self.reset_function()


class LearningRateFinder:
    """
    Learning Rate Finder for optimal learning rate discovery.

    Implements the learning rate range test by training with exponentially
    increasing learning rates and analyzing the loss curve to suggest
    optimal learning rate ranges.

    Parameters
    ----------
    trainer : BaseTrainer
        Training interface object
    min_lr : float, default=1e-7
        Minimum learning rate to test
    max_lr : float, default=10.0
        Maximum learning rate to test
    num_iterations : int, default=100
        Number of iterations to run the test
    step_mode : str, default='exp'
        How to step learning rate ('exp' for exponential, 'linear' for linear)
    smooth_beta : float, default=0.98
        Smoothing factor for loss smoothing (exponential moving average)
    divergence_threshold : float, default=4.0
        Stop if loss > divergence_threshold * min_loss
    """

    def __init__(
        self,
        trainer: BaseTrainer,
        min_lr: float = 1e-7,
        max_lr: float = 10.0,
        num_iterations: int = 100,
        step_mode: str = "exp",
        smooth_beta: float = 0.98,
        divergence_threshold: float = 4.0,
    ):

        self.trainer = trainer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_iterations = num_iterations
        self.step_mode = step_mode.lower()
        self.smooth_beta = smooth_beta
        self.divergence_threshold = divergence_threshold

        # Validate parameters
        if min_lr >= max_lr:
            raise ValueError("min_lr must be less than max_lr")
        if not 0 < smooth_beta < 1:
            raise ValueError("smooth_beta must be between 0 and 1")
        if step_mode not in ["exp", "linear"]:
            raise ValueError("step_mode must be 'exp' or 'linear'")

    def _generate_learning_rates(self) -> np.ndarray:
        """
        Generate learning rate schedule.

        Returns
        -------
        np.ndarray
            Array of learning rates to test
        """
        if self.step_mode == "exp":
            return np.logspace(
                np.log10(self.min_lr), np.log10(self.max_lr), self.num_iterations
            )
        else:
            return np.linspace(self.min_lr, self.max_lr, self.num_iterations)

    def _smooth_losses(self, losses: np.ndarray) -> np.ndarray:
        """
        Apply exponential smoothing to losses.

        Parameters
        ----------
        losses : np.ndarray
            Raw loss values

        Returns
        -------
        np.ndarray
            Smoothed loss values
        """
        smoothed = np.zeros_like(losses)
        smoothed[0] = losses[0]

        for i in range(1, len(losses)):
            smoothed[i] = (
                self.smooth_beta * smoothed[i - 1] + (1 - self.smooth_beta) * losses[i]
            )

        return smoothed

    def _detect_divergence(self, losses: np.ndarray, iteration: int) -> bool:
        """
        Detect if training has diverged.

        Parameters
        ----------
        losses : np.ndarray
            Loss values so far
        iteration : int
            Current iteration

        Returns
        -------
        bool
            True if divergence detected
        """
        if iteration < 10:  # Need some history
            return False

        min_loss = np.min(losses[: iteration + 1])
        current_loss = losses[iteration]

        return current_loss > self.divergence_threshold * min_loss

    def _analyze_results(
        self,
        learning_rates: np.ndarray,
        losses: np.ndarray,
        smoothed_losses: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Analyze learning rate finder results.

        Parameters
        ----------
        learning_rates : np.ndarray
            Learning rates tested
        losses : np.ndarray
            Raw loss values
        smoothed_losses : np.ndarray
            Smoothed loss values

        Returns
        -------
        dict
            Analysis results and metrics
        """
        analysis = {}

        analysis["min_loss"] = float(np.min(losses))
        analysis["max_loss"] = float(np.max(losses))
        analysis["min_loss_lr"] = float(learning_rates[np.argmin(losses)])

        gradients = np.gradient(smoothed_losses, np.log10(learning_rates))
        min_gradient_idx = np.argmin(gradients)
        analysis["min_gradient_lr"] = float(learning_rates[min_gradient_idx])
        analysis["min_gradient"] = float(gradients[min_gradient_idx])

        suggested_lr = analysis["min_gradient_lr"] / 10
        analysis["suggested_lr"] = float(suggested_lr)

        if len(losses) > 10:
            initial_loss = np.mean(losses[:5])
            min_loss = analysis["min_loss"]
            analysis["loss_reduction_ratio"] = (initial_loss - min_loss) / initial_loss
        else:
            analysis["loss_reduction_ratio"] = 0.0

        loss_variance = np.var(losses)
        analysis["loss_variance"] = float(loss_variance)
        analysis["coefficient_of_variation"] = float(
            np.sqrt(loss_variance) / np.mean(losses)
        )

        if len(losses) > 20:
            first_half_mean = np.mean(losses[: len(losses) // 2])
            second_half_mean = np.mean(losses[len(losses) // 2 :])
            analysis["convergence_ratio"] = second_half_mean / first_half_mean
        else:
            analysis["convergence_ratio"] = 1.0

        return analysis

    def find(self, verbose: bool = True) -> LearningRateFinderResult:
        """
        Run learning rate finder.

        Parameters
        ----------
        verbose : bool, default=True
            Whether to print progress information

        Returns
        -------
        LearningRateFinderResult
            Results of the learning rate finder
        """
        if verbose:
            print("Starting Learning Rate Finder...")
            print(f"Learning rate range: {self.min_lr:.2e} to {self.max_lr:.2e}")
            print(f"Number of iterations: {self.num_iterations}")
            print(f"Step mode: {self.step_mode}")

        learning_rates = self._generate_learning_rates()
        losses = []

        self.trainer.reset_model()

        for i, lr in enumerate(learning_rates):
            try:
                loss = self.trainer.train_batch(lr)

                if np.isnan(loss) or np.isinf(loss):
                    if verbose:
                        print(
                            f"Iteration {i + 1}: Learning rate {lr:.2e} - Invalid loss (nan/inf)"
                        )
                    break

                losses.append(float(loss))

                if self._detect_divergence(np.array(losses), i):
                    if verbose:
                        print(
                            f"Iteration {i + 1}: Learning rate {lr:.2e} - Training diverged"
                        )
                    break

                if verbose and (i + 1) % max(1, self.num_iterations // 10) == 0:
                    print(
                        f"Iteration {i + 1}/{self.num_iterations}: LR = {lr:.2e}, Loss = {loss:.6f}"
                    )

            except Exception as e:
                if verbose:
                    print(
                        f"Iteration {i + 1}: Learning rate {lr:.2e} - Training failed: {e}"
                    )
                break

        if len(losses) < 5:
            raise RuntimeError(
                "Learning rate finder failed - insufficient valid loss values"
            )

        learning_rates = learning_rates[: len(losses)]
        losses = np.array(losses)

        smoothed_losses = self._smooth_losses(losses)

        analysis = self._analyze_results(learning_rates, losses, smoothed_losses)

        if verbose:
            print(f"\nLearning Rate Finder completed!")
            print(f"Suggested learning rate: {analysis['suggested_lr']:.2e}")
            print(
                f"Learning rate with steepest gradient: {analysis['min_gradient_lr']:.2e}"
            )
            print(f"Minimum loss: {analysis['min_loss']:.6f}")
            print(f"Loss reduction ratio: {analysis['loss_reduction_ratio']:.2%}")

        return LearningRateFinderResult(
            learning_rates=learning_rates,
            losses=losses,
            smoothed_losses=smoothed_losses,
            suggested_lr=analysis["suggested_lr"],
            min_gradient_lr=analysis["min_gradient_lr"],
            analysis=analysis,
        )

    def plot_results(
        self,
        result: LearningRateFinderResult,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot learning rate finder results.

        Parameters
        ----------
        result : LearningRateFinderResult
            Results from learning rate finder
        figsize : tuple, default=(12, 8)
            Figure size for the plot
        save_path : str, optional
            Path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # Plot 1: Learning rate vs Loss (log scale)
        ax1.semilogx(
            result.learning_rates, result.losses, "b-", alpha=0.7, label="Raw Loss"
        )
        ax1.semilogx(
            result.learning_rates,
            result.smoothed_losses,
            "r-",
            linewidth=2,
            label="Smoothed Loss",
        )
        ax1.axvline(
            result.suggested_lr,
            color="green",
            linestyle="--",
            alpha=0.8,
            label=f"Suggested LR: {result.suggested_lr:.2e}",
        )
        ax1.axvline(
            result.min_gradient_lr,
            color="orange",
            linestyle="--",
            alpha=0.8,
            label=f"Min Gradient LR: {result.min_gradient_lr:.2e}",
        )
        ax1.set_xlabel("Learning Rate")
        ax1.set_ylabel("Loss")
        ax1.set_title("Learning Rate vs Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Learning rate vs Loss (linear scale, zoomed)
        # Focus on the interesting region around minimum
        min_loss_idx = np.argmin(result.smoothed_losses)
        start_idx = max(0, min_loss_idx - 20)
        end_idx = min(len(result.learning_rates), min_loss_idx + 20)

        ax2.plot(
            result.learning_rates[start_idx:end_idx],
            result.losses[start_idx:end_idx],
            "b-",
            alpha=0.7,
        )
        ax2.plot(
            result.learning_rates[start_idx:end_idx],
            result.smoothed_losses[start_idx:end_idx],
            "r-",
            linewidth=2,
        )
        ax2.axvline(result.suggested_lr, color="green", linestyle="--", alpha=0.8)
        ax2.axvline(result.min_gradient_lr, color="orange", linestyle="--", alpha=0.8)
        ax2.set_xlabel("Learning Rate")
        ax2.set_ylabel("Loss")
        ax2.set_title("Loss (Zoomed Region)")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Loss gradient
        gradients = np.gradient(result.smoothed_losses, np.log10(result.learning_rates))
        ax3.semilogx(result.learning_rates, gradients, "purple", linewidth=2)
        ax3.axvline(result.min_gradient_lr, color="orange", linestyle="--", alpha=0.8)
        ax3.axhline(0, color="black", linestyle="-", alpha=0.3)
        ax3.set_xlabel("Learning Rate")
        ax3.set_ylabel("Loss Gradient")
        ax3.set_title("Loss Gradient vs Learning Rate")
        ax3.grid(True, alpha=0.3)

        # Plot 4: Statistics summary
        ax4.axis("off")
        stats_text = f"""
        Analysis Summary:
        
        Suggested Learning Rate: {result.suggested_lr:.2e}
        Min Gradient Learning Rate: {result.min_gradient_lr:.2e}
        
        Minimum Loss: {result.analysis["min_loss"]:.6f}
        Loss Reduction: {result.analysis["loss_reduction_ratio"]:.2%}
        
        Loss Variance: {result.analysis["loss_variance"]:.6f}
        Coefficient of Variation: {result.analysis["coefficient_of_variation"]:.3f}
        
        Convergence Ratio: {result.analysis["convergence_ratio"]:.3f}
        """
        ax4.text(
            0.05,
            0.95,
            stats_text,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()


def find_learning_rate(
    train_function: Callable[[float], float],
    reset_function: Callable[[], None],
    min_lr: float = 1e-7,
    max_lr: float = 10.0,
    num_iterations: int = 100,
    step_mode: str = "exp",
    smooth_beta: float = 0.98,
    verbose: bool = True,
    plot: bool = True,
) -> LearningRateFinderResult:
    """
    Convenience function to find optimal learning rate.

    Parameters
    ----------
    train_function : callable
        Function that takes learning_rate (float) and returns loss (float)
    reset_function : callable
        Function to reset model to initial state
    min_lr : float, default=1e-7
        Minimum learning rate to test
    max_lr : float, default=10.0
        Maximum learning rate to test
    num_iterations : int, default=100
        Number of iterations for the test
    step_mode : str, default='exp'
        Learning rate stepping mode ('exp' or 'linear')
    smooth_beta : float, default=0.98
        Smoothing factor for loss curves
    verbose : bool, default=True
        Whether to print progress
    plot : bool, default=True
        Whether to plot results

    Returns
    -------
    LearningRateFinderResult
        Results including suggested learning rate

    Examples
    --------
    >>> # Example with simple quadratic loss
    >>> def train_step(lr):
    ...     # Simulate one training step
    ...     current_w = getattr(train_step, "w", 1.0)  # Get current weight
    ...     target_w = 0.5  # Target weight
    ...     loss = (current_w - target_w) ** 2
    ...
    ...     # Gradient descent update
    ...     gradient = 2 * (current_w - target_w)
    ...     train_step.w = current_w - lr * gradient
    ...
    ...     return loss + np.random.normal(0, 0.01)  # Add noise
    >>> def reset_model():
    ...     train_step.w = 1.0  # Reset to initial weight
    >>> result = find_learning_rate(
    ...     train_step, reset_model, min_lr=1e-4, max_lr=1.0, num_iterations=50
    ... )
    >>> print(f"Suggested learning rate: {result.suggested_lr}")
    """
    trainer = FunctionTrainer(train_function, reset_function)

    finder = LearningRateFinder(
        trainer=trainer,
        min_lr=min_lr,
        max_lr=max_lr,
        num_iterations=num_iterations,
        step_mode=step_mode,
        smooth_beta=smooth_beta,
    )

    result = finder.find(verbose=verbose)

    if plot:
        finder.plot_results(result)

    return result


def suggest_learning_rate_schedule(
    result: LearningRateFinderResult, schedule_type: str = "onecycle"
) -> Dict[str, Any]:
    """
    Suggest learning rate schedule based on finder results.

    Parameters
    ----------
    result : LearningRateFinderResult
        Results from learning rate finder
    schedule_type : str, default='onecycle'
        Type of schedule to suggest ('onecycle', 'cyclic', 'cosine', 'step')

    Returns
    -------
    dict
        Suggested schedule parameters
    """
    max_lr = result.min_gradient_lr
    base_lr = result.suggested_lr

    if schedule_type == "onecycle":
        return {
            "schedule_type": "onecycle",
            "max_lr": max_lr,
            "base_lr": base_lr,
            "pct_start": 0.3,  # 30% warmup
            "final_div_factor": 1e4,  # Final LR = max_lr / final_div_factor
            "description": "One-cycle policy with warmup and annealing",
        }

    elif schedule_type == "cyclic":
        return {
            "schedule_type": "cyclic",
            "base_lr": base_lr,
            "max_lr": max_lr,
            "step_size_up": 2000,  # Steps to increase from base to max
            "mode": "triangular2",  # Decreasing amplitude
            "description": "Cyclical learning rate with triangular policy",
        }

    elif schedule_type == "cosine":
        return {
            "schedule_type": "cosine",
            "initial_lr": max_lr,
            "min_lr": base_lr,
            "T_max": 10,  # Period of cosine annealing
            "description": "Cosine annealing with restarts",
        }

    elif schedule_type == "step":
        return {
            "schedule_type": "step",
            "initial_lr": max_lr / 3,  # Conservative start
            "step_size": 10,  # Epochs between reductions
            "gamma": 0.5,  # Multiplication factor
            "description": "Step decay schedule",
        }

    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


if __name__ == "__main__":
    # Example 1: Simple quadratic optimization
    print("Example 1: Simple Quadratic Function")
    print("Optimizing f(w) = (w - 0.5)^2 with gradient descent")

    def quadratic_train_step(lr):
        """Simulate one training step on quadratic function."""
        current_w = getattr(quadratic_train_step, "w", 1.0)
        target_w = 0.5

        loss = (current_w - target_w) ** 2
        gradient = 2 * (current_w - target_w)

        quadratic_train_step.w = current_w - lr * gradient

        # Add some noise to simulate realistic training
        return loss + np.random.normal(0, 0.001)

    def reset_quadratic():
        """Reset model to initial state."""
        quadratic_train_step.w = 1.0

    result1 = find_learning_rate(
        train_function=quadratic_train_step,
        reset_function=reset_quadratic,
        min_lr=1e-3,
        max_lr=2.0,
        num_iterations=50,
        verbose=True,
        plot=True,
    )

    print(f"Suggested learning rate: {result1.suggested_lr:.4f}")
    print(f"Learning rate with steepest gradient: {result1.min_gradient_lr:.4f}")

    # Example 2: Neural network simulation
    print("\n" + "=" * 60)
    print("Example 2: Simulated Neural Network Training")

    class SimulatedNN:
        """Simulated neural network for demonstration."""

        def __init__(self):
            self.reset()

        def reset(self):
            """Reset network to initial state."""
            self.weights = np.random.normal(0, 1, 10)  # 10 random weights
            self.momentum = np.zeros_like(self.weights)
            self.step_count = 0

        def train_step(self, learning_rate):
            """Simulate one training step."""
            self.step_count += 1

            # Simulate loss function with multiple minima
            base_loss = 2.0 * np.exp(-self.step_count / 20)

            # Learning rate effects
            if learning_rate < 1e-5:
                lr_penalty = 0.5  # Too small - slow convergence
            elif learning_rate > 0.1:
                lr_penalty = (learning_rate - 0.1) * 10  # Too large - instability
            else:
                lr_penalty = 0.0

            gradients = np.random.normal(0, 0.1, len(self.weights))
            self.weights -= learning_rate * gradients

            # Total loss with noise
            loss = base_loss + lr_penalty + np.random.normal(0, 0.01)
            return max(loss, 0.001)

    sim_nn = SimulatedNN()

    result2 = find_learning_rate(
        train_function=sim_nn.train_step,
        reset_function=sim_nn.reset,
        min_lr=1e-6,
        max_lr=1.0,
        num_iterations=80,
        verbose=True,
        plot=False,
    )

    print(f"Suggested learning rate: {result2.suggested_lr:.2e}")
    print(f"Learning rate with steepest gradient: {result2.min_gradient_lr:.2e}")

    print("\nSuggested Learning Rate Schedules:")
    for schedule_type in ["onecycle", "cyclic", "cosine", "step"]:
        schedule = suggest_learning_rate_schedule(result2, schedule_type)
        print(f"\n{schedule_type.upper()}:")
        print(f"  Description: {schedule['description']}")
        for key, value in schedule.items():
            if key != "description":
                if isinstance(value, float):
                    print(f"  {key}: {value:.2e}")
                else:
                    print(f"  {key}: {value}")

    # Example 3: Advanced analysis
    print("\n" + "=" * 60)
    print("Example 3: Advanced Analysis")

    step_modes = ["exp", "linear"]
    results = {}

    for mode in step_modes:
        sim_nn.reset()
        result = find_learning_rate(
            train_function=sim_nn.train_step,
            reset_function=sim_nn.reset,
            min_lr=1e-5,
            max_lr=0.5,
            num_iterations=60,
            step_mode=mode,
            verbose=False,
            plot=False,
        )
        results[mode] = result
        print(f"\n{mode.upper()} stepping mode:")
        print(f"  Suggested LR: {result.suggested_lr:.2e}")
        print(f"  Min gradient LR: {result.min_gradient_lr:.2e}")
        print(f"  Loss reduction: {result.analysis['loss_reduction_ratio']:.2%}")
        print(
            f"  Coefficient of variation: {result.analysis['coefficient_of_variation']:.3f}"
        )

    print(f"\nComparison:")
    exp_result = results["exp"]
    lin_result = results["linear"]
    print(f"  Exponential stepping found LR: {exp_result.suggested_lr:.2e}")
    print(f"  Linear stepping found LR: {lin_result.suggested_lr:.2e}")
    difference = (
        abs(exp_result.suggested_lr - lin_result.suggested_lr) / exp_result.suggested_lr
    )
    print(f"  Difference: {difference:.1%}")

    print("\nLearning Rate Finder examples completed!")
    print("In practice, use plot=True to visualize the results.")
