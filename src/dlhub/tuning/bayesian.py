"""
Bayesian Optimization for Hyperparameter Tuning
================================================

Implements Bayesian optimization using Gaussian Process surrogate models with
Expected Improvement acquisition function for efficient hyperparameter search.
This approach is particularly effective for expensive black-box optimization
problems like neural network hyperparameter tuning.

References
----------
- Snoek, J., Larochelle, H., & Adams, R. P. (2012). "Practical Bayesian
  Optimization of Machine Learning Algorithms." NIPS.
- Mockus, J. (1994). "Application of Bayesian approach to numerical methods
  of global and stochastic optimization." Journal of Global Optimization.

Author
------
Deep Learning Reference Hub

License
-------
MIT License

Notes
-----
This implementation uses a simplified Gaussian Process with RBF kernel.
For production use, consider libraries like Optuna, GPyOpt, or scikit-optimize
which provide more robust implementations with additional features.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


@dataclass
class BayesianOptimizationResult:
    """
    Container for Bayesian optimization results.

    Attributes
    ----------
    best_params : dict
        Best hyperparameter configuration found
    best_score : float
        Best objective function value achieved
    history : list
        History of all evaluations
    convergence_data : dict
        Convergence statistics and diagnostics
    """

    best_params: dict[str, Any]
    best_score: float
    history: list[tuple[dict[str, Any], float]]
    convergence_data: dict[str, Any]


class GaussianProcess:
    """
    Simplified Gaussian Process for Bayesian Optimization.

    Implements a GP with RBF kernel for modeling the objective function.
    This is a educational implementation - production code should use
    more robust libraries like GPy or scikit-learn.

    Parameters
    ----------
    kernel_lengthscale : float, default=1.0
        Length scale parameter for RBF kernel
    kernel_variance : float, default=1.0
        Variance parameter for RBF kernel
    noise_variance : float, default=1e-6
        Noise variance for numerical stability
    """

    def __init__(
        self,
        kernel_lengthscale: float = 1.0,
        kernel_variance: float = 1.0,
        noise_variance: float = 1e-6,
    ):
        self.kernel_lengthscale = kernel_lengthscale
        self.kernel_variance = kernel_variance
        self.noise_variance = noise_variance
        self.X_train = None
        self.y_train = None
        self.K_inv = None

    def rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Compute RBF (Radial Basis Function) kernel matrix.

        Parameters
        ----------
        X1 : np.ndarray, shape (n1, d)
            First set of input points
        X2 : np.ndarray, shape (n2, d)
            Second set of input points

        Returns
        -------
        np.ndarray, shape (n1, n2)
            Kernel matrix K(X1, X2)
        """
        sq_dists = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2)

        return self.kernel_variance * np.exp(
            -0.5 * sq_dists / (self.kernel_lengthscale**2)
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Gaussian Process to training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training input points
        y : np.ndarray, shape (n_samples,)
            Training target values
        """
        self.X_train = X.copy()
        self.y_train = y.copy()

        # Compute kernel matrix and its inverse
        K = self.rbf_kernel(X, X)
        K += self.noise_variance * np.eye(len(X))

        try:
            self.K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            warnings.warn("Kernel matrix is singular, using pseudo-inverse")
            self.K_inv = np.linalg.pinv(K)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.

        Parameters
        ----------
        X : np.ndarray, shape (n_test, n_features)
            Test input points

        Returns
        -------
        mean : np.ndarray, shape (n_test,)
            Predicted mean values
        std : np.ndarray, shape (n_test,)
            Predicted standard deviations
        """
        if self.X_train is None:
            raise ValueError("GP must be fitted before making predictions")

        # Compute kernel matrices
        K_star = self.rbf_kernel(X, self.X_train)
        K_star_star = self.rbf_kernel(X, X)

        # Compute predictive mean
        mean = K_star @ self.K_inv @ self.y_train  # type: ignore

        var = np.diag(K_star_star) - np.diag(K_star @ self.K_inv @ K_star.T)
        var = np.maximum(var, 1e-10)
        std = np.sqrt(var)

        return mean, std


class BayesianOptimizer:
    """
    Bayesian Optimization using Gaussian Process surrogate models.

    This implementation uses Expected Improvement as the acquisition function
    to balance exploration and exploitation in hyperparameter search.

    Parameters
    ----------
    objective_function : callable
        Function to optimize. Should take hyperparameter dict and return float
    search_space : dict
        Dictionary defining search space for each hyperparameter.
        Format: {'param_name': (min_val, max_val)} for continuous parameters
    acquisition : str, default='ei'
        Acquisition function ('ei' for Expected Improvement, 'ucb' for UCB)
    kappa : float, default=2.576
        Exploration parameter for UCB (ignored if acquisition='ei')
    xi : float, default=0.01
        Exploration parameter for Expected Improvement
    n_initial : int, default=5
        Number of random initial evaluations
    random_state : int, optional
        Random seed for reproducibility
    """

    def __init__(
        self,
        objective_function: Callable[[dict], float],
        search_space: dict[str, tuple[float, float]],
        acquisition: str = "ei",
        kappa: float = 2.576,
        xi: float = 0.01,
        n_initial: int = 5,
        random_state: int | None = None,
    ):

        self.objective_function = objective_function
        self.search_space = search_space
        self.acquisition = acquisition.lower()
        self.kappa = kappa
        self.xi = xi
        self.n_initial = n_initial

        # Checked here rather than where the acquisition is first evaluated,
        # which is after the initial design has run. Each of those evaluations
        # is a full training run, so a misspelled name must not cost them.
        if self.acquisition not in ("ei", "ucb"):
            raise ValueError(
                f"Unknown acquisition function: {acquisition!r}. Use 'ei' or 'ucb'."
            )

        if random_state is not None:
            np.random.seed(random_state)

        # Initialize internal state
        self.param_names = list(search_space.keys())
        self.bounds = np.array([search_space[name] for name in self.param_names])
        self.gp = GaussianProcess()
        self.X_observed = []
        self.y_observed = []
        self.history = []
        self.best_score = -np.inf
        self.best_params = None

    def _normalize_params(self, X: np.ndarray) -> np.ndarray:
        """Normalize parameters to [0, 1] range."""
        return (X - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])

    def _denormalize_params(self, X_norm: np.ndarray) -> np.ndarray:
        """Denormalize parameters from [0, 1] to original range."""
        return X_norm * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]

    def _array_to_dict(self, X: np.ndarray) -> dict[str, float]:
        """Convert parameter array to dictionary."""
        return {name: float(val) for name, val in zip(self.param_names, X)}

    def _expected_improvement(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Expected Improvement acquisition function.

        Parameters
        ----------
        X : np.ndarray, shape (n_points, n_params)
            Normalized parameter points to evaluate

        Returns
        -------
        np.ndarray, shape (n_points,)
            Expected improvement values
        """
        if len(self.X_observed) == 0:
            return np.ones(len(X))

        mean, std = self.gp.predict(X)

        f_max = max(self.y_observed)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z = (mean - f_max - self.xi) / std
            ei = (mean - f_max - self.xi) * norm.cdf(z) + std * norm.pdf(z)
            ei[std == 0.0] = 0.0

        return ei

    def _upper_confidence_bound(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Upper Confidence Bound acquisition function.

        Parameters
        ----------
        X : np.ndarray, shape (n_points, n_params)
            Normalized parameter points to evaluate

        Returns
        -------
        np.ndarray, shape (n_points,)
            Upper confidence bound values
        """
        if len(self.X_observed) == 0:
            return np.ones(len(X))

        mean, std = self.gp.predict(X)
        return mean + self.kappa * std

    def _acquisition_function(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the chosen acquisition function."""
        if self.acquisition == "ei":
            return self._expected_improvement(X)
        elif self.acquisition == "ucb":
            return self._upper_confidence_bound(X)
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition}")

    def _optimize_acquisition(self) -> np.ndarray:
        """
        Find the point that maximizes the acquisition function.

        Returns
        -------
        np.ndarray, shape (n_params,)
            Normalized parameters that maximize acquisition function
        """

        # Objective to minimize (negative acquisition)
        def objective(x):
            return -self._acquisition_function(x.reshape(1, -1))[0]

        # Try multiple random starting points
        n_restarts = 10
        best_x = None
        best_val = np.inf

        for _ in range(n_restarts):
            x0 = np.random.uniform(0, 1, len(self.param_names))

            try:
                result = minimize(
                    objective,
                    x0,
                    bounds=[(0, 1)] * len(self.param_names),
                    method="L-BFGS-B",
                )

                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x
            except Exception:
                continue

        if best_x is None:
            best_x = np.random.uniform(0, 1, len(self.param_names))

        return best_x

    def _evaluate_objective(self, params: dict[str, float]) -> float:
        """
        Evaluate objective function and handle exceptions.

        Parameters
        ----------
        params : dict
            Hyperparameter configuration

        Returns
        -------
        float
            Objective function value (np.nan if evaluation failed)
        """
        try:
            score = self.objective_function(params)
            if np.isnan(score) or np.isinf(score):
                return np.nan
            return float(score)
        except Exception as e:
            warnings.warn(f"Objective evaluation failed: {e}")
            return np.nan

    def optimize(
        self, n_iterations: int = 20, verbose: int = 1
    ) -> BayesianOptimizationResult:
        """
        Run Bayesian optimization.

        Parameters
        ----------
        n_iterations : int, default=20
            Maximum number of optimization iterations
        verbose : int, default=1
            Reporting level: 0 is silent, 1 prints phases and new bests, 2 adds
            every evaluation and the final configuration.

        Returns
        -------
        BayesianOptimizationResult
            Optimization results including best parameters and history
        """
        if verbose >= 1:
            print("Starting Bayesian Optimization...")
            print(f"Search space: {self.search_space}")
            print(f"Acquisition function: {self.acquisition}")

        if verbose >= 1:
            print(f"\nPhase 1: Random initialization ({self.n_initial} points)")

        for i in range(self.n_initial):
            X_raw = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            params = self._array_to_dict(X_raw)

            score = self._evaluate_objective(params)

            if not np.isnan(score):
                X_norm = self._normalize_params(X_raw)
                self.X_observed.append(X_norm)
                self.y_observed.append(score)
                self.history.append((params.copy(), score))

                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()

                if verbose >= 1:
                    print(f"  {i + 1}/{self.n_initial}: Score = {score:.4f}")

        if len(self.X_observed) == 0:
            raise RuntimeError("All initial evaluations failed")

        if verbose >= 1:
            print(f"\nPhase 2: Bayesian optimization ({n_iterations} iterations)")

        for iteration in range(n_iterations):
            X_train = np.array(self.X_observed)
            y_train = np.array(self.y_observed)
            self.gp.fit(X_train, y_train)

            X_next_norm = self._optimize_acquisition()
            X_next_raw = self._denormalize_params(X_next_norm)
            params_next = self._array_to_dict(X_next_raw)

            score = self._evaluate_objective(params_next)

            if not np.isnan(score):
                self.X_observed.append(X_next_norm)
                self.y_observed.append(score)
                self.history.append((params_next.copy(), score))

                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params_next.copy()

                    if verbose >= 1:
                        print(
                            f"  Iter {iteration + 1}: Score = {score:.4f} (NEW BEST!)"
                        )
                else:
                    if verbose >= 2:
                        print(f"  Iter {iteration + 1}: Score = {score:.4f}")
            else:
                if verbose >= 1:
                    print(f"  Iter {iteration + 1}: Evaluation failed")

        successful_initial = [s for _, s in self.history[: self.n_initial]]
        convergence_data = {
            "n_evaluations": len(self.history),
            "n_failed": n_iterations + self.n_initial - len(self.history),
            # The initial design is random, so its mean is what an equal budget
            # of random search would have averaged. Guarded because every
            # initial point can fail while later ones succeed, and a mean over
            # an empty list is a nan that propagates into the whole summary.
            "improvement_over_random": (
                self.best_score - float(np.mean(successful_initial))
                if successful_initial
                else 0.0
            ),
            "scores": [score for _, score in self.history],
        }

        if verbose >= 1:
            print("\nOptimization completed!")
            print(f"Best score: {self.best_score:.4f}")
        if verbose >= 2:
            print(f"Best parameters: {self.best_params}")
            print(f"Total evaluations: {len(self.history)}")

        return BayesianOptimizationResult(
            best_params=self.best_params,  # type: ignore
            best_score=self.best_score,
            history=self.history,
            convergence_data=convergence_data,
        )


def bayesian_optimize(
    objective_function: Callable[[dict], float],
    search_space: dict[str, tuple[float, float]],
    n_iterations: int = 20,
    n_initial: int = 5,
    acquisition: str = "ei",
    random_state: int | None = None,
    verbose: int = 1,
) -> BayesianOptimizationResult:
    """
    Run a Bayesian search over continuous bounds, returning its own result type.

    The method-specific entry point, alongside :func:`~dlhub.tuning.random_search`
    and :func:`~dlhub.tuning.asha_optimize`. To choose a method by name instead,
    call :func:`~dlhub.tuning.optimize_hyperparameters`, which takes the
    framework's richer search-space format and returns an ``ExperimentResult``.

    Takes a scalar objective and a ``{name: (low, high)}`` search space; every
    dimension is continuous, since the Gaussian process interpolates over them.

    Parameters
    ----------
    objective_function : callable
        Function to optimize. Should accept hyperparameter dict and return float
    search_space : dict
        Search space definition: {'param_name': (min_val, max_val)}
    n_iterations : int, default=20
        Number of optimization iterations after initial random sampling
    n_initial : int, default=5
        Number of random initial points
    acquisition : str, default='ei'
        Acquisition function ('ei' or 'ucb')
    random_state : int, optional
        Random seed for reproducibility
    verbose : int, default=1
        Reporting level, as in :meth:`BayesianOptimizer.optimize`

    Returns
    -------
    BayesianOptimizationResult
        Optimization results

    Examples
    --------
    >>> def objective(params):
    ...     # Simulate training a model and return validation accuracy
    ...     lr, wd = params["learning_rate"], params["weight_decay"]
    ...     # Dummy objective (replace with actual model training)
    ...     return -((lr - 0.001) ** 2) - (wd - 0.0001) ** 2 + np.random.normal(0, 0.01)
    >>>
    >>> search_space = {"learning_rate": (1e-5, 1e-1), "weight_decay": (1e-6, 1e-2)}
    >>>
    >>> result = bayesian_optimize(
    ...     objective, search_space, n_iterations=30, random_state=42
    ... )
    >>> print(f"Best parameters: {result.best_params}")
    """
    optimizer = BayesianOptimizer(
        objective_function=objective_function,
        search_space=search_space,
        acquisition=acquisition,
        n_initial=n_initial,
        random_state=random_state,
    )

    return optimizer.optimize(n_iterations=n_iterations, verbose=verbose)


if __name__ == "__main__":

    def quadratic_objective(params):
        """Example objective function - quadratic with noise."""
        x, y = params["x"], params["y"]
        # Global minimum at (2, -1) with value -5
        return -((x - 2) ** 2) - (y + 1) ** 2 - 5 + np.random.normal(0, 0.1)

    search_space: dict[str, tuple[float, float]] = {"x": (-5, 5), "y": (-5, 5)}

    print("Example: Optimizing quadratic function")
    print("True optimum: x=2, y=-1, value=-5")

    result = bayesian_optimize(
        objective_function=quadratic_objective,
        search_space=search_space,
        n_iterations=25,
        n_initial=5,
        random_state=42,
        verbose=True,
    )

    print(f"\nFound optimum: {result.best_params}")
    print(f"Best value: {result.best_score:.3f}")

    def nn_objective(params):
        """Dummy neural network training objective."""
        lr = params["learning_rate"]
        batch_size = int(params["batch_size"])
        hidden_units = int(params["hidden_units"])

        optimal_lr = 0.003
        optimal_batch = 64
        optimal_hidden = 256

        # Penalty terms for deviations from optimal values
        lr_penalty = -10 * (np.log10(lr) - np.log10(optimal_lr)) ** 2
        batch_penalty = -0.001 * (batch_size - optimal_batch) ** 2
        hidden_penalty = -0.00001 * (hidden_units - optimal_hidden) ** 2

        # Base performance + penalties + noise
        performance = 0.95 + lr_penalty + batch_penalty + hidden_penalty
        performance += np.random.normal(0, 0.02)  # Training noise

        return performance

    nn_search_space = {
        "learning_rate": (1e-5, 1e-1),
        "batch_size": (16, 256),
        "hidden_units": (64, 512),
    }

    print("\n" + "=" * 60)
    print("Example: Neural Network Hyperparameter Optimization")

    nn_result = bayesian_optimize(
        objective_function=nn_objective,
        search_space=nn_search_space,
        n_iterations=30,
        n_initial=8,
        random_state=42,
    )

    print(f"\nOptimal hyperparameters found:")
    for param, value in nn_result.best_params.items():
        print(f"  {param}: {value:.6f}")
    print(f"Best validation accuracy: {nn_result.best_score:.4f}")
