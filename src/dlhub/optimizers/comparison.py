"""
Optimization Algorithms Comparison
==================================

A comprehensive comparison framework for evaluating different optimization algorithms
in deep learning contexts. This module provides implementations and utilities to
compare gradient descent variants (SGD, Momentum, RMSprop, Adam) on various loss
landscapes and datasets, demonstrating their convergence properties and performance
characteristics.

This implementation serves as both a practical tool for optimizer selection and
an educational resource for understanding optimization dynamics in neural networks.

References
----------
- Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization.
  arXiv preprint arXiv:1412.6980.
- Ruder, S. (2016). An overview of gradient descent optimization algorithms.
  arXiv preprint arXiv:1609.04747.
- Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods for
  online learning and stochastic optimization. JMLR, 12, 2121-2159.

Author
------
Deep Learning Reference Hub

License
-------
MIT License

Notes
-----
This implementation focuses on numerical stability and educational clarity.
The optimizers raced here are the canonical implementations from this
subpackage, presented through a uniform driver interface rather than
reimplemented; see :data:`RACE_BIAS_CORRECTION` for the one configuration
decision the race makes on their behalf.
Visualization utilities require matplotlib and are designed for Jupyter notebooks.
"""

import time
import warnings
from dataclasses import dataclass
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

from dlhub.optimizers.adam import AdamOptimizer as CanonicalAdam
from dlhub.optimizers.base import BaseOptimizer
from dlhub.optimizers.mini_batch import MiniBatchGradientDescent
from dlhub.optimizers.momentum import MomentumOptimizer as CanonicalMomentum
from dlhub.optimizers.rmsprop import RMSpropOptimizer as CanonicalRMSprop

warnings.filterwarnings("ignore", category=RuntimeWarning)

#: Whether the race asks for bias correction wherever an optimizer supports it.
#:
#: The question is unavoidable and was previously answered by accident. Adam
#: bias-corrects unconditionally, by construction; on Momentum and RMSprop it is
#: a choice, and the canonical modules make it a constructor flag with different
#: defaults -- on for Momentum, off for RMSprop, each matching how that method is
#: usually written.
#:
#: Racing them at their own defaults would put a bias-corrected Momentum against
#: an uncorrected RMSprop against a corrected Adam, so part of any gap between
#: the curves would be the correction rather than the method. The race therefore
#: asks for it everywhere it is available, which is also what the harness did
#: before it delegated: its own copies applied the correction unconditionally.
#:
#: This governs the race only. Each adapter takes ``bias_correction`` and passes
#: it through, so a caller comparing against a canonical module's own output can
#: ask for that module's default instead.
RACE_BIAS_CORRECTION = True


class OptimizerType(Enum):
    """Enumeration of available optimizer types."""

    SGD = "sgd"
    MOMENTUM = "momentum"
    RMSPROP = "rmsprop"
    ADAM = "adam"


@dataclass
class OptimizationRun:
    """
    The trace of one optimizer descending one problem, and its summary.

    Not to be confused with :class:`dlhub.tuning.ExperimentResult`, which records
    the outcome of a hyperparameter search rather than a single descent.

    Attributes
    ----------
    optimizer_name : str
        Name of the optimizer used.
    losses : List[float]
        Loss values recorded during training.
    parameters : List[Dict]
        Parameter values at each iteration.
    convergence_time : float
        Time taken for convergence (in seconds).
    final_loss : float
        Final loss value achieved.
    iterations_to_converge : int
        Number of iterations required for convergence.
    """

    optimizer_name: str
    losses: list[float]
    parameters: list[dict]
    convergence_time: float
    final_loss: float
    iterations_to_converge: int


class SGDOptimizer(BaseOptimizer):
    """
    Stochastic Gradient Descent optimizer.

    Basic gradient descent with fixed learning rate. Simple but often effective
    baseline for comparison with more sophisticated optimizers.

    The update rule itself lives in :mod:`dlhub.optimizers.mini_batch`, which is
    where the hub teaches plain gradient descent; this class exists to present it
    through the driver contract so the race can hold it alongside the others.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for parameter updates.
    """

    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate)
        self.name = "SGD"
        self._optimizer = MiniBatchGradientDescent(learning_rate=learning_rate)

    def update_parameters(self, params: dict, grads: dict, t: int) -> dict:
        """
        Update parameters using vanilla gradient descent.

        Parameters
        ----------
        params : dict
            Current parameter values.
        grads : dict
            Gradients for each parameter.
        t : int
            Current iteration. Unused: the step is the same at every iteration,
            since plain gradient descent carries no state to correct for.

        Returns
        -------
        dict
            Updated parameters after SGD step.
        """
        return self._optimizer.update_parameters(params, grads)


class MomentumOptimizer(BaseOptimizer):
    """
    Momentum optimizer using exponential moving averages.

    Accelerates gradient descent by accumulating momentum in consistent directions
    and dampening oscillations. Particularly effective in ravines and saddle points.

    Wraps :class:`dlhub.optimizers.momentum.MomentumOptimizer`, which is where the
    update rule is derived and written out.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for parameter updates.
    beta : float, default=0.9
        Momentum coefficient for exponential moving average.
    bias_correction : bool, default=``RACE_BIAS_CORRECTION``
        Whether to divide out the bias of the zero-initialised velocity. See
        :data:`RACE_BIAS_CORRECTION` for why the race sets it rather than
        inheriting the canonical module's own default.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        beta: float = 0.9,
        bias_correction: bool = RACE_BIAS_CORRECTION,
    ):
        super().__init__(learning_rate)
        self.beta = beta
        self.bias_correction = bias_correction
        self.name = "Momentum"
        self._optimizer = CanonicalMomentum(
            learning_rate=learning_rate, beta=beta, bias_correction=bias_correction
        )

    def update_parameters(self, params: dict, grads: dict, t: int) -> dict:
        """
        Update parameters using momentum-based gradient descent.

        Parameters
        ----------
        params : dict
            Current parameter values.
        grads : dict
            Gradients for each parameter.
        t : int
            Current iteration. Unused: the canonical optimizer counts its own
            steps, and :meth:`reset` clears that count, so the two agree for any
            driver that counts from one and resets between runs.

        Returns
        -------
        dict
            Updated parameters after momentum step.
        """
        return self._optimizer.update_parameters(params, grads)

    def reset(self) -> None:
        """Reset momentum terms for new optimization run."""
        self._optimizer.reset_optimizer_state()


class RMSpropOptimizer(BaseOptimizer):
    """
    RMSprop (Root Mean Square Propagation) optimizer.

    Adapts the learning rate per parameter using a running average of squared
    gradients, so that parameters with consistently large gradients take smaller
    steps.

    Wraps :class:`dlhub.optimizers.rmsprop.RMSpropOptimizer`, which is where the
    update rule is derived and written out.

    Parameters
    ----------
    learning_rate : float, default=0.001
        Step size for parameter updates.
    beta : float, default=0.9
        Decay rate for the running average of squared gradients.
    epsilon : float, default=1e-8
        Small constant to prevent division by zero.
    bias_correction : bool, default=``RACE_BIAS_CORRECTION``
        Whether to divide out the bias of the zero-initialised second moment.
        The canonical module defaults this off, which is how RMSprop is usually
        written; see :data:`RACE_BIAS_CORRECTION` for why the race overrides it.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta: float = 0.9,
        epsilon: float = 1e-8,
        bias_correction: bool = RACE_BIAS_CORRECTION,
    ):
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.bias_correction = bias_correction
        self.name = "RMSprop"
        self._optimizer = CanonicalRMSprop(
            learning_rate=learning_rate,
            beta=beta,
            epsilon=epsilon,
            bias_correction=bias_correction,
        )

    def update_parameters(self, params: dict, grads: dict, t: int) -> dict:
        """
        Update parameters using RMSprop optimization.

        Parameters
        ----------
        params : dict
            Current parameter values.
        grads : dict
            Gradients for each parameter.
        t : int
            Current iteration. Unused, for the reason given on
            :meth:`MomentumOptimizer.update_parameters`.

        Returns
        -------
        dict
            Updated parameters after RMSprop step.
        """
        return self._optimizer.update_parameters(params, grads)

    def reset(self) -> None:
        """Reset second moment estimates for new optimization run."""
        self._optimizer.reset_optimizer_state()


class AdamOptimizer(BaseOptimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer.

    Combines benefits of Momentum and RMSprop by maintaining both first and second
    moment estimates of gradients. Generally robust and effective across many problems.

    Wraps :class:`dlhub.optimizers.adam.AdamOptimizer`, which is where the update
    rule is derived and written out. Adam bias-corrects both moments by
    construction, so it takes no flag: the correction is part of the method
    rather than an option on it.

    Parameters
    ----------
    learning_rate : float, default=0.001
        Step size for parameter updates.
    beta1 : float, default=0.9
        Exponential decay rate for first moment estimates.
    beta2 : float, default=0.999
        Exponential decay rate for second moment estimates.
    epsilon : float, default=1e-8
        Small constant to prevent division by zero.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.name = "Adam"
        self._optimizer = CanonicalAdam(
            learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon
        )

    def update_parameters(self, params: dict, grads: dict, t: int) -> dict:
        """
        Update parameters using Adam optimization algorithm.

        Parameters
        ----------
        params : dict
            Current parameter values.
        grads : dict
            Gradients for each parameter.
        t : int
            Current iteration. Unused, for the reason given on
            :meth:`MomentumOptimizer.update_parameters`.

        Returns
        -------
        dict
            Updated parameters after Adam step.
        """
        return self._optimizer.update(params, grads)

    def reset(self) -> None:
        """Reset first and second moment estimates for new optimization run."""
        self._optimizer.reset_state()


class OptimizationProblem:
    """
    Base class for defining optimization problems with loss functions and gradients.

    Parameters
    ----------
    name : str
        Name of the optimization problem.
    """

    def __init__(self, name: str):
        self.name = name

    def loss_function(self, params: dict) -> float:
        """
        Compute loss for given parameters.

        Parameters
        ----------
        params : dict
            Parameter values.

        Returns
        -------
        float
            Loss value.
        """
        raise NotImplementedError("Subclasses must implement loss_function")

    def gradients(self, params: dict) -> dict:
        """
        Compute gradients for given parameters.

        Parameters
        ----------
        params : dict
            Parameter values.

        Returns
        -------
        dict
            Gradients for each parameter.
        """
        raise NotImplementedError("Subclasses must implement gradients")

    def initial_parameters(self) -> dict:
        """
        Get initial parameter values for optimization.

        Returns
        -------
        dict
            Initial parameter values.
        """
        raise NotImplementedError("Subclasses must implement initial_parameters")


class QuadraticBowl(OptimizationProblem):
    """
    Simple quadratic bowl optimization problem: f(x,y) = ax² + by².

    Well-conditioned convex problem useful for demonstrating basic optimizer behavior.

    Parameters
    ----------
    a : float, default=1.0
        Coefficient for x² term.
    b : float, default=1.0
        Coefficient for y² term.
    """

    def __init__(self, a: float = 1.0, b: float = 1.0):
        super().__init__(f"Quadratic Bowl (a={a}, b={b})")
        self.a = a
        self.b = b

    def loss_function(self, params: dict) -> float:
        """Compute quadratic loss: ax² + by²."""
        x, y = params["x"], params["y"]
        return self.a * x**2 + self.b * y**2

    def gradients(self, params: dict) -> dict:
        """Compute gradients: [2ax, 2by]."""
        x, y = params["x"], params["y"]
        return {"x": 2 * self.a * x, "y": 2 * self.b * y}

    def initial_parameters(self) -> dict:
        """Initialize at (5, 5) for clear visualization."""
        return {"x": 5.0, "y": 5.0}


class RosenbrockFunction(OptimizationProblem):
    """
    Rosenbrock function: f(x,y) = (a-x)² + b(y-x²)².

    Classic non-convex optimization benchmark with narrow curved valley.
    Challenging for optimizers due to ill-conditioning and plateau regions.

    Parameters
    ----------
    a : float, default=1.0
        Parameter controlling x-offset of minimum.
    b : float, default=100.0
        Parameter controlling valley curvature (higher = more challenging).
    """

    def __init__(self, a: float = 1.0, b: float = 100.0):
        super().__init__(f"Rosenbrock Function (a={a}, b={b})")
        self.a = a
        self.b = b

    def loss_function(self, params: dict) -> float:
        """Compute Rosenbrock function value."""
        x, y = params["x"], params["y"]
        return (self.a - x) ** 2 + self.b * (y - x**2) ** 2

    def gradients(self, params: dict) -> dict:
        """Compute Rosenbrock gradients analytically."""
        x, y = params["x"], params["y"]
        dx = -2 * (self.a - x) - 4 * self.b * x * (y - x**2)
        dy = 2 * self.b * (y - x**2)
        return {"x": dx, "y": dy}

    def initial_parameters(self) -> dict:
        """Initialize away from minimum for interesting optimization path."""
        return {"x": -2.0, "y": 2.0}


class BealeFunction(OptimizationProblem):
    """
    Beale function: f(x,y) = (1.5 - x + xy)² + (2.25 - x + x*y²)² + (2.625 - x + x*y³)².

    Multimodal function with global minimum and several local minima.
    Tests optimizer robustness to local minima and saddle points.
    """

    def __init__(self):
        super().__init__("Beale Function")

    def loss_function(self, params: dict) -> float:
        """Compute Beale function value."""
        x, y = params["x"], params["y"]
        term1 = (1.5 - x + x * y) ** 2
        term2 = (2.25 - x + x * y**2) ** 2
        term3 = (2.625 - x + x * y**3) ** 2
        return term1 + term2 + term3

    def gradients(self, params: dict) -> dict:
        """Compute Beale function gradients analytically."""
        x, y = params["x"], params["y"]

        # Partial derivatives computed analytically
        dx = (
            2 * (1.5 - x + x * y) * (-1 + y)
            + 2 * (2.25 - x + x * y**2) * (-1 + y**2)
            + 2 * (2.625 - x + x * y**3) * (-1 + y**3)
        )

        dy = (
            2 * (1.5 - x + x * y) * x
            + 2 * (2.25 - x + x * y**2) * (2 * x * y)
            + 2 * (2.625 - x + x * y**3) * (3 * x * y**2)
        )

        return {"x": dx, "y": dy}

    def initial_parameters(self) -> dict:
        """Initialize at challenging starting point."""
        return {"x": 4.0, "y": 4.0}


class OptimizationComparison:
    """
    Comprehensive framework for comparing optimization algorithms.

    Provides utilities to run multiple optimizers on various problems,
    collect performance metrics, and generate comparative visualizations.

    Parameters
    ----------
    max_iterations : int, default=1000
        Maximum number of optimization iterations.
    tolerance : float, default=1e-6
        Convergence tolerance for loss change.
    verbose : bool, default=True
        Whether to print progress information.
    """

    def __init__(
        self, max_iterations: int = 1000, tolerance: float = 1e-6, verbose: bool = True
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.results = {}

    def create_optimizer(
        self, optimizer_type: OptimizerType, **kwargs
    ) -> BaseOptimizer:
        """
        Factory method to create optimizer instances.

        Parameters
        ----------
        optimizer_type : OptimizerType
            Type of optimizer to create.
        **kwargs
            Additional parameters for optimizer initialization.

        Returns
        -------
        BaseOptimizer
            Configured optimizer instance.
        """
        if optimizer_type == OptimizerType.SGD:
            return SGDOptimizer(**kwargs)
        elif optimizer_type == OptimizerType.MOMENTUM:
            return MomentumOptimizer(**kwargs)
        elif optimizer_type == OptimizerType.RMSPROP:
            return RMSpropOptimizer(**kwargs)
        elif optimizer_type == OptimizerType.ADAM:
            return AdamOptimizer(**kwargs)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    def run_optimization(
        self, problem: OptimizationProblem, optimizer: BaseOptimizer
    ) -> OptimizationRun:
        """
        Run single optimization experiment.

        Parameters
        ----------
        problem : OptimizationProblem
            Problem to optimize.
        optimizer : BaseOptimizer
            Optimizer to use.

        Returns
        -------
        OptimizationRun
            Results of optimization including metrics and trajectory.
        """
        optimizer.reset()

        # The benchmark problems below are two-dimensional surfaces, so they state
        # their starting points as plain floats, which reads naturally for a point
        # on a contour plot. Optimizers are written against arrays -- the contract
        # says so, and a neural network's parameters are never scalars -- so the
        # conversion happens once, here, where the problem's world meets the
        # optimizer's. Doing it at the boundary means every optimizer receives what
        # the contract promises, rather than each one having to tolerate floats.
        params = {
            name: np.asarray(value, dtype=float)
            for name, value in problem.initial_parameters().items()
        }
        losses = []
        parameter_history = []

        start_time = time.time()
        converged = False

        for iteration in range(1, self.max_iterations + 1):
            current_loss = problem.loss_function(params)
            grads = problem.gradients(params)

            losses.append(current_loss)
            parameter_history.append(params.copy())

            if len(losses) > 1:
                loss_change = abs(losses[-2] - losses[-1])
                if loss_change < self.tolerance:
                    converged = True
                    break

            params = optimizer.update_parameters(params, grads, iteration)

            # Prevent divergence
            if current_loss > 1e10 or np.any(
                [np.isnan(v) or np.isinf(v) for v in params.values()]
            ):
                if self.verbose:
                    print(f"{optimizer.name} diverged at iteration {iteration}")
                break

        end_time = time.time()

        result = OptimizationRun(
            optimizer_name=optimizer.name,
            losses=losses,
            parameters=parameter_history,
            convergence_time=end_time - start_time,
            final_loss=losses[-1] if losses else float("inf"),
            iterations_to_converge=len(losses) if converged else self.max_iterations,
        )

        if self.verbose:
            status = "converged" if converged else "max iterations reached"
            print(
                f"{optimizer.name}: {status} in {len(losses)} iterations, "
                f"final loss: {result.final_loss:.6f}, time: {result.convergence_time:.3f}s"
            )

        return result

    def compare_optimizers(
        self, problem: OptimizationProblem, optimizer_configs: dict[OptimizerType, dict]
    ) -> dict[str, OptimizationRun]:
        """
        Compare multiple optimizers on a single problem.

        Parameters
        ----------
        problem : OptimizationProblem
            Problem to optimize.
        optimizer_configs : dict
            Dictionary mapping optimizer types to their configuration parameters.

        Returns
        -------
        dict
            Results for each optimizer.
        """
        results = {}

        if self.verbose:
            print(f"\nOptimizing: {problem.name}")
            print("=" * 50)

        for opt_type, config in optimizer_configs.items():
            optimizer = self.create_optimizer(opt_type, **config)
            result = self.run_optimization(problem, optimizer)
            results[optimizer.name] = result

        return results

    def run_comprehensive_comparison(self) -> dict[str, dict[str, OptimizationRun]]:
        """
        Run comprehensive comparison across multiple problems and optimizers.

        Returns
        -------
        dict
            Nested dictionary: {problem_name: {optimizer_name: result}}
        """
        problems = [
            QuadraticBowl(a=1.0, b=1.0),  # Well-conditioned
            QuadraticBowl(a=1.0, b=100.0),  # Ill-conditioned
            RosenbrockFunction(),  # Non-convex valley
            BealeFunction(),  # Multimodal
        ]

        optimizer_configs = {
            OptimizerType.SGD: {"learning_rate": 0.01},
            OptimizerType.MOMENTUM: {"learning_rate": 0.01, "beta": 0.9},
            OptimizerType.RMSPROP: {"learning_rate": 0.01, "beta": 0.9},
            OptimizerType.ADAM: {"learning_rate": 0.05, "beta1": 0.9, "beta2": 0.999},
        }

        all_results = {}

        for problem in problems:
            results = self.compare_optimizers(problem, optimizer_configs)
            all_results[problem.name] = results

        self.results = all_results
        return all_results

    def plot_convergence_comparison(
        self,
        results: dict[str, OptimizationRun],
        problem_name: str,
        log_scale: bool = True,
    ):
        """
        Plot convergence curves for optimizer comparison.

        Parameters
        ----------
        results : dict
            Results from optimizer comparison.
        problem_name : str
            Name of the problem for plot title.
        log_scale : bool, default=True
            Whether to use logarithmic scale for loss.
        """
        plt.figure(figsize=(12, 8))

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        styles = ["-", "--", "-.", ":"]

        for i, (opt_name, result) in enumerate(results.items()):
            if result.losses:
                iterations = range(1, len(result.losses) + 1)
                plt.plot(
                    iterations,
                    result.losses,
                    color=colors[i % len(colors)],
                    linestyle=styles[i % len(styles)],
                    linewidth=2,
                    label=opt_name,
                    alpha=0.8,
                )

        plt.xlabel("Iterations", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title(
            f"Convergence Comparison: {problem_name}", fontsize=14, fontweight="bold"
        )
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        if log_scale:
            plt.yscale("log")
            plt.ylabel("Loss (log scale)", fontsize=12)

        plt.tight_layout()
        plt.show()

    def plot_optimization_paths(
        self,
        results: dict[str, OptimizationRun],
        problem: OptimizationProblem,
        contour_levels: int = 20,
    ):
        """
        Plot optimization trajectories on loss landscape contours.

        Parameters
        ----------
        results : dict
            Results from optimizer comparison.
        problem : OptimizationProblem
            Problem instance for computing loss landscape.
        contour_levels : int, default=20
            Number of contour levels to display.
        """
        plt.figure(figsize=(14, 10))

        # Create meshgrid for contour plot
        x_range = np.linspace(-6, 6, 100)
        y_range = np.linspace(-6, 6, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                params = {"x": X[i, j], "y": Y[i, j]}
                Z[i, j] = problem.loss_function(params)

        contours = plt.contour(
            X, Y, Z, levels=contour_levels, alpha=0.6, cmap="viridis"
        )
        plt.clabel(contours, inline=True, fontsize=8, fmt="%.1f")

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        markers = ["o", "s", "^", "D"]

        for i, (opt_name, result) in enumerate(results.items()):
            if result.parameters:
                x_path = [p["x"] for p in result.parameters]
                y_path = [p["y"] for p in result.parameters]

                plt.plot(
                    x_path,
                    y_path,
                    color=colors[i % len(colors)],
                    marker=markers[i % len(markers)],
                    markersize=4,
                    linewidth=2,
                    label=f"{opt_name} ({len(x_path)} steps)",
                    alpha=0.8,
                )

                # Mark start and end points
                plt.plot(
                    x_path[0],
                    y_path[0],
                    marker="*",
                    color=colors[i % len(colors)],
                    markersize=12,
                    markeredgecolor="black",
                    markeredgewidth=1,
                )
                plt.plot(
                    x_path[-1],
                    y_path[-1],
                    marker="x",
                    color=colors[i % len(colors)],
                    markersize=10,
                    markeredgewidth=3,
                )

        plt.xlabel("x", fontsize=12)
        plt.ylabel("y", fontsize=12)
        plt.title(f"Optimization Paths: {problem.name}", fontsize=14, fontweight="bold")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        plt.tight_layout()
        plt.show()

    def generate_summary_table(
        self, all_results: dict[str, dict[str, OptimizationRun]]
    ) -> None:
        """
        Generate formatted summary table of optimization results.

        Parameters
        ----------
        all_results : dict
            Complete results from comprehensive comparison.
        """
        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPARISON SUMMARY")
        print("=" * 80)

        for problem_name, results in all_results.items():
            print(f"\n{problem_name}")
            print("-" * len(problem_name))

            # Create formatted table
            headers = [
                "Optimizer",
                "Final Loss",
                "Iterations",
                "Conv. Time (s)",
                "Status",
            ]
            print(
                f"{headers[0]:<12} {headers[1]:<20} {headers[2]:<12} {headers[3]:<15} {headers[4]:<10}"
            )
            print("-" * 80)

            for opt_name, result in results.items():
                status = (
                    "✓"
                    if result.iterations_to_converge < self.max_iterations
                    else "Max iter"
                )
                print(
                    f"{opt_name:<12} {result.final_loss:<14.6f} "
                    f"{result.iterations_to_converge:<12} "
                    f"{result.convergence_time:<15.3f} {status:<10}"
                )

        print("\n" + "=" * 90)


def main():
    """
    Main function demonstrating comprehensive optimization comparison.

    Runs all optimizers on multiple test problems and generates
    comparative visualizations and summary statistics.
    """
    print("🚀 Starting Comprehensive Optimization Algorithm Comparison")
    print("=" * 60)

    comparator = OptimizationComparison(
        max_iterations=500, tolerance=1e-8, verbose=True
    )
    all_results = comparator.run_comprehensive_comparison()
    comparator.generate_summary_table(all_results)

    problems = [
        QuadraticBowl(a=1.0, b=1.0),
        QuadraticBowl(a=1.0, b=100.0),
        RosenbrockFunction(),
        BealeFunction(),
    ]

    print("\n📊 Generating Visualization Plots...")
    for problem in problems:
        if problem.name in all_results:
            results = all_results[problem.name]

            print(f"Plotting convergence for: {problem.name}")
            comparator.plot_convergence_comparison(results, problem.name)

            print(f"Plotting optimization paths for: {problem.name}")
            comparator.plot_optimization_paths(results, problem)

    print("\n✅ Optimization comparison completed successfully!")
    print("Check the generated plots to analyze optimizer performance.")

    return all_results


class OptimizationAnalytics:
    """
    Advanced analytics utilities for optimization comparison results.

    Provides statistical analysis, performance ranking, and detailed
    insights into optimizer behavior across different problem types.
    """

    @staticmethod
    def compute_convergence_metrics(result: OptimizationRun) -> dict[str, float]:
        """
        Compute detailed convergence metrics for a single optimization run.

        Parameters
        ----------
        result : OptimizationRun
            Single optimization result to analyze.

        Returns
        -------
        dict
            Dictionary of computed metrics.
        """
        losses = np.array(result.losses)

        metrics = {
            "final_loss": result.final_loss,
            "initial_loss": losses[0] if len(losses) > 0 else float("inf"),
            "loss_reduction": losses[0] - result.final_loss if len(losses) > 0 else 0.0,
            "relative_improvement": ((losses[0] - result.final_loss) / losses[0]) * 100
            if len(losses) > 0 and losses[0] != 0
            else 0.0,
            "iterations_to_converge": result.iterations_to_converge,
            "convergence_time": result.convergence_time,
            "convergence_rate": 0.0,
            "stability_score": 0.0,
        }

        # Compute convergence rate (loss decrease per iteration)
        if len(losses) > 1:
            total_improvement = losses[0] - losses[-1]
            metrics["convergence_rate"] = total_improvement / len(losses)

            # Compute stability score (1 - coefficient of variation of loss changes)
            loss_changes = np.diff(losses)
            if len(loss_changes) > 0 and np.mean(loss_changes) != 0:
                cv = np.std(loss_changes) / abs(np.mean(loss_changes))
                metrics["stability_score"] = max(0, 1 - cv)

        return metrics

    @staticmethod
    def rank_optimizers(
        all_results: dict[str, dict[str, OptimizationRun]],
    ) -> dict[str, dict[str, int]]:
        """
        Rank optimizers across different problems and metrics.

        Parameters
        ----------
        all_results : dict
            Complete results from optimization comparison.

        Returns
        -------
        dict
            Rankings for each optimizer on each problem.
        """
        rankings = {}

        for problem_name, results in all_results.items():
            if not results:
                continue

            optimizer_metrics = {}
            for opt_name, result in results.items():
                optimizer_metrics[opt_name] = (
                    OptimizationAnalytics.compute_convergence_metrics(result)
                )

            rankings[problem_name] = {}
            sorted_by_loss = sorted(
                optimizer_metrics.items(), key=lambda x: x[1]["final_loss"]
            )
            for rank, (opt_name, _) in enumerate(sorted_by_loss, 1):
                rankings[problem_name][f"{opt_name}_loss_rank"] = rank

            sorted_by_speed = sorted(
                optimizer_metrics.items(), key=lambda x: x[1]["iterations_to_converge"]
            )
            for rank, (opt_name, _) in enumerate(sorted_by_speed, 1):
                rankings[problem_name][f"{opt_name}_speed_rank"] = rank

            sorted_by_stability = sorted(
                optimizer_metrics.items(),
                key=lambda x: x[1]["stability_score"],
                reverse=True,
            )
            for rank, (opt_name, _) in enumerate(sorted_by_stability, 1):
                rankings[problem_name][f"{opt_name}_stability_rank"] = rank

        return rankings

    @staticmethod
    def generate_performance_heatmap(
        all_results: dict[str, dict[str, OptimizationRun]],
    ):
        """
        Generate performance heatmap comparing optimizers across problems.

        Parameters
        ----------
        all_results : dict
            Complete results from optimization comparison.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        optimizers = []
        problems = list(all_results.keys())

        for problem_results in all_results.values():
            for opt_name in problem_results.keys():
                if opt_name not in optimizers:
                    optimizers.append(opt_name)

        performance_matrix = np.zeros((len(optimizers), len(problems)))

        for j, problem_name in enumerate(problems):
            results = all_results[problem_name]
            losses = [results[opt].final_loss for opt in optimizers if opt in results]

            if losses:
                # Normalize losses (0 = best, 1 = worst)
                min_loss, max_loss = min(losses), max(losses)
                loss_range = max_loss - min_loss if max_loss != min_loss else 1

                for i, opt_name in enumerate(optimizers):
                    if opt_name in results:
                        normalized_loss = (
                            results[opt_name].final_loss - min_loss
                        ) / loss_range
                        performance_matrix[i, j] = normalized_loss
                    else:
                        performance_matrix[i, j] = (
                            1.0  # Worst performance if not available
                        )

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(
            performance_matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1
        )

        ax.set_xticks(range(len(problems)))
        ax.set_yticks(range(len(optimizers)))
        ax.set_xticklabels(
            [p.split("(")[0].strip() for p in problems], rotation=45, ha="right"
        )
        ax.set_yticklabels(optimizers)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(
            "Normalized Performance (0=Best, 1=Worst)", rotation=270, labelpad=20
        )

        # Add text annotations
        for i in range(len(optimizers)):
            for j in range(len(problems)):
                text = ax.text(
                    j,
                    i,
                    f"{performance_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                )

        ax.set_title(
            "Optimizer Performance Heatmap Across Problems",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.tight_layout()
        plt.show()


def run_custom_experiment():
    """
    Example of running custom optimization experiment with specific configurations.

    Demonstrates how to use the framework for targeted analysis with
    custom hyperparameters and problems.
    """
    print("\n🔬 Running Custom Optimization Experiment")
    print("=" * 50)

    problem = RosenbrockFunction(a=1.0, b=10.0)  # Less challenging than default

    custom_configs = {
        OptimizerType.SGD: {"learning_rate": 0.001},
        OptimizerType.MOMENTUM: {"learning_rate": 0.001, "beta": 0.95},
        OptimizerType.RMSPROP: {"learning_rate": 0.001, "beta": 0.99},
        OptimizerType.ADAM: {"learning_rate": 0.05, "beta1": 0.9, "beta2": 0.999},
    }

    comparator = OptimizationComparison(
        max_iterations=2000, tolerance=1e-10, verbose=True
    )
    results = comparator.compare_optimizers(problem, custom_configs)

    print("\n📈 Custom Experiment Results:")
    for opt_name, result in results.items():
        metrics = OptimizationAnalytics.compute_convergence_metrics(result)
        print(f"{opt_name}:")
        print(f"  Final Loss: {metrics['final_loss']:.8f}")
        print(f"  Improvement: {metrics['relative_improvement']:.2f}%")
        print(f"  Convergence Rate: {metrics['convergence_rate']:.6f} loss/iter")
        print(f"  Stability Score: {metrics['stability_score']:.3f}")
        print()

    comparator.plot_convergence_comparison(results, "Custom Rosenbrock Experiment")
    comparator.plot_optimization_paths(results, problem)

    return results


def demonstrate_hyperparameter_sensitivity():
    """
    Demonstrate sensitivity of optimizers to hyperparameter choices.

    Shows how different learning rates affect optimizer performance
    on the same problem, highlighting the importance of tuning.
    """
    print("\n⚙️  Hyperparameter Sensitivity Analysis")
    print("=" * 50)

    problem = QuadraticBowl(a=1.0, b=10.0)  # Moderately ill-conditioned
    learning_rates = [0.001, 0.01, 0.1, 0.5]

    plt.figure(figsize=(15, 10))

    for i, optimizer_type in enumerate([OptimizerType.SGD, OptimizerType.ADAM]):
        plt.subplot(2, 2, i * 2 + 1)

        for lr in learning_rates:
            comparator = OptimizationComparison(max_iterations=200, verbose=False)

            if optimizer_type == OptimizerType.SGD:
                optimizer = SGDOptimizer(learning_rate=lr)
            else:
                optimizer = AdamOptimizer(learning_rate=lr)

            result = comparator.run_optimization(problem, optimizer)

            if result.losses:
                plt.plot(result.losses, label=f"LR={lr}", linewidth=2, alpha=0.8)

        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title(f"{optimizer_type.value.upper()} - Learning Rate Sensitivity")
        plt.yscale("log")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Show final loss comparison
        plt.subplot(2, 2, i * 2 + 2)
        final_losses = []

        for lr in learning_rates:
            comparator = OptimizationComparison(max_iterations=200, verbose=False)

            if optimizer_type == OptimizerType.SGD:
                optimizer = SGDOptimizer(learning_rate=lr)
            else:
                optimizer = AdamOptimizer(learning_rate=lr)

            result = comparator.run_optimization(problem, optimizer)
            final_losses.append(result.final_loss)

        plt.bar(range(len(learning_rates)), final_losses, alpha=0.7)
        plt.xlabel("Learning Rate")
        plt.ylabel("Final Loss")
        plt.title(f"{optimizer_type.value.upper()} - Final Loss vs Learning Rate")
        plt.xticks(range(len(learning_rates)), [str(lr) for lr in learning_rates])
        plt.yscale("log")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = main()

    print("\n🎯 Advanced Analytics")
    print("=" * 40)

    OptimizationAnalytics.generate_performance_heatmap(results)

    rankings = OptimizationAnalytics.rank_optimizers(results)
    print("\n🏆 Optimizer Rankings by Problem:")
    for problem, ranks in rankings.items():
        print(f"\n{problem}:")
        optimizers = set(k.split("_")[0] for k in ranks.keys())
        for opt in optimizers:
            loss_rank = ranks.get(f"{opt}_loss_rank", "N/A")
            speed_rank = ranks.get(f"{opt}_speed_rank", "N/A")
            stability_rank = ranks.get(f"{opt}_stability_rank", "N/A")
            print(
                f"  {opt}: Loss={loss_rank}, Speed={speed_rank}, Stability={stability_rank}"
            )

    custom_results = run_custom_experiment()
    demonstrate_hyperparameter_sensitivity()

    print("\n🎉 All experiments completed! Analysis summary:")
    print("- Adam generally shows robust performance across problems")
    print("- Momentum helps with ill-conditioned problems")
    print("- RMSprop adapts well to different loss landscapes")
    print("- SGD requires careful learning rate tuning")
    print("- Problem characteristics significantly affect optimizer choice")
