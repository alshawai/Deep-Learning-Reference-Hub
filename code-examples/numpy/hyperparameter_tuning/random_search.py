"""
Random Search for Hyperparameter Tuning
========================================

Comprehensive random search implementation with proper probability distributions
and parallel evaluation support. Random search has been shown to be more effective
than grid search for high-dimensional hyperparameter optimization problems.

References
----------
- Bergstra, J., & Bengio, Y. (2012). "Random search for hyper-parameter
  optimization." Journal of Machine Learning Research, 13, 281-305.
- Li, L., et al. (2017). "Hyperband: A novel bandit-based approach to
  hyperparameter optimization." Journal of Machine Learning Research, 18, 1-52.

Author
------
Deep Learning Reference Hub

License
-------
MIT License

Notes
-----
Random search is particularly effective when only a few hyperparameters matter
for the final performance. It explores the hyperparameter space more efficiently
than grid search by sampling more unique values per dimension.
"""

import numpy as np
from scipy.stats import uniform, loguniform, randint
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import dataclass, field
import time
import warnings


@dataclass
class RandomSearchResult:
    """
    Container for random search optimization results.

    Attributes
    ----------
    best_params : dict
        Best hyperparameter configuration found
    best_score : float
        Best objective function value achieved
    all_params : list
        All parameter configurations evaluated
    all_scores : list
        All scores corresponding to parameter configurations
    search_time : float
        Total search time in seconds
    statistics : dict
        Search statistics and analysis
    """

    best_params: Dict[str, Any]
    best_score: float
    all_params: List[Dict[str, Any]]
    all_scores: List[float]
    search_time: float
    statistics: Dict[str, Any] = field(default_factory=dict)


class ParameterDistribution:
    """
    Base class for hyperparameter distributions.

    Defines the interface for sampling hyperparameters from different
    probability distributions.
    """

    def sample(self) -> Any:
        """Sample a value from the distribution."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class UniformDistribution(ParameterDistribution):
    """
    Uniform distribution for continuous parameters.

    Parameters
    ----------
    low : float
        Lower bound of the distribution
    high : float
        Upper bound of the distribution
    """

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high
        self._dist = uniform(loc=low, scale=high - low)

    def sample(self) -> float:
        """Sample from uniform distribution."""
        return self._dist.rvs()

    def __repr__(self) -> str:
        return f"UniformDistribution(low={self.low}, high={self.high})"


class LogUniformDistribution(ParameterDistribution):
    """
    Log-uniform distribution for parameters that vary over orders of magnitude.

    Particularly useful for learning rates, regularization parameters, etc.

    Parameters
    ----------
    low : float
        Lower bound (must be positive)
    high : float
        Upper bound (must be positive)
    """

    def __init__(self, low: float, high: float):
        if low <= 0 or high <= 0:
            raise ValueError("Log-uniform distribution requires positive bounds")
        self.low = low
        self.high = high
        self._dist = loguniform(a=low, b=high)

    def sample(self) -> float:
        """Sample from log-uniform distribution."""
        return self._dist.rvs()

    def __repr__(self) -> str:
        return f"LogUniformDistribution(low={self.low}, high={self.high})"


class IntegerDistribution(ParameterDistribution):
    """
    Discrete uniform distribution for integer parameters.

    Parameters
    ----------
    low : int
        Lower bound (inclusive)
    high : int
        Upper bound (exclusive)
    """

    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high
        self._dist = randint(low=low, high=high)

    def sample(self) -> int:
        """Sample from discrete uniform distribution."""
        return int(self._dist.rvs())

    def __repr__(self) -> str:
        return f"IntegerDistribution(low={self.low}, high={self.high})"


class ChoiceDistribution(ParameterDistribution):
    """
    Categorical distribution for discrete choices.

    Parameters
    ----------
    choices : list
        List of possible values to choose from
    probabilities : list, optional
        Probability weights for each choice (uniform if None)
    """

    def __init__(self, choices: List[Any], probabilities: Optional[List[float]] = None):
        self.choices = choices
        if probabilities is None:
            self.probabilities = [1.0 / len(choices)] * len(choices)
        else:
            if len(probabilities) != len(choices):
                raise ValueError("Probabilities must match number of choices")
            # Normalize probabilities
            total = sum(probabilities)
            self.probabilities = [p / total for p in probabilities]

    def sample(self) -> Any:
        """Sample from categorical distribution."""
        return np.random.choice(self.choices, p=self.probabilities)

    def __repr__(self) -> str:
        return f"ChoiceDistribution(choices={self.choices})"


class PowerDistribution(ParameterDistribution):
    """
    Power law distribution for parameters with non-uniform preferences.

    Useful when smaller values are preferred (common in regularization).

    Parameters
    ----------
    low : float
        Lower bound
    high : float
        Upper bound
    power : float, default=2.0
        Power parameter (higher values favor smaller numbers)
    """

    def __init__(self, low: float, high: float, power: float = 2.0):
        self.low = low
        self.high = high
        self.power = power

    def sample(self) -> float:
        """Sample from power distribution."""
        u = np.random.random()
        return self.low + (self.high - self.low) * (u ** (1.0 / self.power))

    def __repr__(self) -> str:
        return (
            f"PowerDistribution(low={self.low}, high={self.high}, power={self.power})"
        )


class RandomSearchOptimizer:
    """
    Random Search optimizer for hyperparameter tuning.

    Implements efficient random sampling of hyperparameters with support for
    different probability distributions, parallel evaluation, and early stopping.

    Parameters
    ----------
    objective_function : callable
        Function to optimize. Should take hyperparameter dict and return float
    search_space : dict
        Dictionary mapping parameter names to ParameterDistribution objects
    n_iter : int, default=100
        Number of parameter configurations to sample and evaluate
    random_state : int (optional)
        Random seed for reproducibility
    n_jobs : int, default=1
        Number of parallel jobs (-1 for all available cores)
    early_stopping : bool, default=False
        Whether to use early stopping based on improvement
    patience : int, default=10
        Number of iterations without improvement before stopping
    """

    def __init__(
        self,
        objective_function: Callable[[Dict], float],
        search_space: Dict[str, ParameterDistribution],
        n_iter: int = 100,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
        early_stopping: bool = False,
        patience: int = 10,
    ) -> None:

        self.objective_function = objective_function
        self.search_space = search_space
        self.n_iter = n_iter
        self.random_state = random_state
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.early_stopping = early_stopping
        self.patience = patience

        if random_state is not None:
            np.random.seed(random_state)

        self.results_history = []
        self.best_score = -np.inf
        self.best_params = None
        self.iterations_without_improvement = 0

    def sample_parameters(self) -> Dict[str, Any]:
        """
        Sample a single parameter configuration from the search space.

        Returns
        -------
        dict
            Sampled hyperparameter configuration
        """
        params = {}
        for param_name, distribution in self.search_space.items():
            params[param_name] = distribution.sample()
        return params

    def sample_multiple_parameters(self, n_samples: int) -> List[Dict[str, Any]]:
        """
        Sample multiple parameter configurations.

        Parameters
        ----------
        n_samples : int
            Number of configurations to sample

        Returns
        -------
        list
            List of parameter configurations
        """
        return [self.sample_parameters() for _ in range(n_samples)]

    def _evaluate_single(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """
        Evaluate objective function for a single parameter configuration.

        Parameters
        ----------
        params : dict
            Parameter configuration to evaluate

        Returns
        -------
        tuple
            (parameters, score) tuple
        """
        try:
            score = self.objective_function(params)
            if np.isnan(score) or np.isinf(score):
                return params, -np.inf
            return params, float(score)
        except Exception as e:
            warnings.warn(f"Evaluation failed for {params}: {e}")
            return params, -np.inf

    def _evaluate_batch_sequential(
        self, param_list: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Evaluate parameters sequentially."""
        results = []
        for params in param_list:
            result = self._evaluate_single(params)
            results.append(result)
        return results

    def _evaluate_batch_parallel(
        self, param_list: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Evaluate parameters in parallel."""
        results = []
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit all jobs
            future_to_params = {
                executor.submit(self._evaluate_single, params): params
                for params in param_list
            }

            # Collect results as they complete
            for future in as_completed(future_to_params):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    params = future_to_params[future]
                    warnings.warn(f"Parallel evaluation failed for {params}: {e}")
                    results.append((params, -np.inf))

        return results

    def _should_stop_early(self) -> bool:
        """Check if early stopping criteria are met."""
        if not self.early_stopping:
            return False
        return self.iterations_without_improvement >= self.patience

    def optimize(self, verbose: bool = True) -> RandomSearchResult:
        """
        Run random search optimization.

        Parameters
        ----------
        verbose : bool, default=True
            Whether to print progress information

        Returns
        -------
        RandomSearchResult
            Optimization results
        """
        if verbose:
            print("Starting Random Search Optimization...")
            print(f"Search space: {len(self.search_space)} parameters")
            print(f"Number of iterations: {self.n_iter}")
            print(f"Parallel jobs: {self.n_jobs}")
            if self.early_stopping:
                print(f"Early stopping: patience={self.patience}")

        start_time = time.time()
        all_params = []
        all_scores = []

        if self.n_jobs > 1:
            batch_size = min(self.n_jobs * 2, self.n_iter)
        else:
            batch_size = 10  # Process in small batches even for sequential

        iterations_completed = 0

        while iterations_completed < self.n_iter:
            remaining_iterations = self.n_iter - iterations_completed
            current_batch_size = min(batch_size, remaining_iterations)

            param_batch = self.sample_multiple_parameters(current_batch_size)

            if self.n_jobs > 1:
                batch_results = self._evaluate_batch_parallel(param_batch)
            else:
                batch_results = self._evaluate_batch_sequential(param_batch)

            improvement_found = False
            for params, score in batch_results:
                all_params.append(params)
                all_scores.append(score)

                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
                    improvement_found = True
                    self.iterations_without_improvement = 0

                    if verbose:
                        print(
                            f"Iteration {iterations_completed + 1}: "
                            f"New best score = {score:.6f}"
                        )
                else:
                    self.iterations_without_improvement += 1

                iterations_completed += 1

                if self._should_stop_early():
                    if verbose:
                        print(f"Early stopping at iteration {iterations_completed}")
                    break

            if self._should_stop_early():
                break

            # Progress update
            if verbose and iterations_completed % max(1, self.n_iter // 10) == 0:
                print(
                    f"Progress: {iterations_completed}/{self.n_iter} "
                    f"({100 * iterations_completed / self.n_iter:.1f}%) "
                    f"- Best score: {self.best_score:.6f}"
                )

        search_time = time.time() - start_time

        valid_scores = [s for s in all_scores if s != -np.inf]
        statistics = {
            "total_evaluations": len(all_scores),
            "successful_evaluations": len(valid_scores),
            "failed_evaluations": len(all_scores) - len(valid_scores),
            "mean_score": np.mean(valid_scores) if valid_scores else -np.inf,
            "std_score": np.std(valid_scores) if len(valid_scores) > 1 else 0.0,
            "score_percentiles": {
                "25th": np.percentile(valid_scores, 25) if valid_scores else -np.inf,
                "50th": np.percentile(valid_scores, 50) if valid_scores else -np.inf,
                "75th": np.percentile(valid_scores, 75) if valid_scores else -np.inf,
                "95th": np.percentile(valid_scores, 95) if valid_scores else -np.inf,
            },
            "improvement_over_random": (self.best_score - np.mean(valid_scores[:10]))
            if len(valid_scores) >= 10
            else 0.0,
            "early_stopped": self._should_stop_early(),
            "iterations_completed": iterations_completed,
        }

        if verbose:
            print(f"\nOptimization completed in {search_time:.2f} seconds!")
            print(f"Best score: {self.best_score:.6f}")
            print(f"Best parameters: {self.best_params}")
            print(f"Total evaluations: {statistics['total_evaluations']}")
            print(
                f"Success rate: {statistics['successful_evaluations'] / statistics['total_evaluations']:.2%}"
            )

        return RandomSearchResult(
            best_params=self.best_params,  # type: ignore
            best_score=self.best_score,
            all_params=all_params,
            all_scores=all_scores,
            search_time=search_time,
            statistics=statistics,
        )


def random_search(
    objective_function: Callable[[Dict], float],
    search_space: Dict[str, Union[ParameterDistribution, Tuple, List]],
    n_iter: int = 100,
    random_state: Optional[int] = None,
    n_jobs: int = 1,
    early_stopping: bool = False,
    patience: int = 10,
    verbose: bool = True,
) -> RandomSearchResult:
    """
    Convenience function for random search hyperparameter optimization.

    Parameters
    ----------
    objective_function : callable
        Function to optimize. Should accept hyperparameter dict and return float
    search_space : dict
        Search space definition. Can contain:
        - ParameterDistribution objects
        - Tuples (min, max) for uniform distributions
        - Lists for categorical choices
    n_iter : int, default=100
        Number of parameter configurations to evaluate
    random_state : int (optional)
        Random seed for reproducibility
    n_jobs : int, default=1
        Number of parallel jobs (-1 for all cores)
    early_stopping : bool, default=False
        Whether to use early stopping
    patience : int, default=10
        Early stopping patience
    verbose : bool, default=True
        Whether to print progress

    Returns
    -------
    RandomSearchResult
        Optimization results

    Examples
    --------
    >>> def objective(params):
    ...     lr, wd = params["learning_rate"], params["weight_decay"]
    ...     return -((lr - 0.001) ** 2) - (wd - 0.0001) ** 2 + np.random.normal(0, 0.01)
    >>> # Simple tuple/list format
    >>> search_space = {
    ...     "learning_rate": (1e-5, 1e-1),  # Will use log-uniform
    ...     "batch_size": [16, 32, 64, 128],  # Will use choice
    ...     "hidden_units": (64, 512),  # Will use uniform
    ... }
    >>> result = random_search(objective, search_space, n_iter=50, random_state=42)
    >>> # Advanced distribution format
    >>> from scipy.stats import truncnorm
    >>> search_space_advanced = {
    ...     "learning_rate": LogUniformDistribution(1e-5, 1e-1),
    ...     "weight_decay": PowerDistribution(1e-6, 1e-2, power=2.0),
    ...     "batch_size": ChoiceDistribution([16, 32, 64, 128], [0.1, 0.3, 0.4, 0.2]),
    ...     "dropout_rate": UniformDistribution(0.0, 0.5),
    ... }
    """
    # Convert simple formats to distribution objects
    processed_space = {}
    for param_name, param_spec in search_space.items():
        if isinstance(param_spec, ParameterDistribution):
            processed_space[param_name] = param_spec
        elif isinstance(param_spec, (tuple, list)) and len(param_spec) == 2:
            low, high = param_spec
            if isinstance(low, (int, float)) and isinstance(high, (int, float)):
                if low > 0 and high / low > 100:  # Use log-uniform for wide ranges
                    processed_space[param_name] = LogUniformDistribution(low, high)
                elif isinstance(low, int) and isinstance(high, int):
                    processed_space[param_name] = IntegerDistribution(low, high + 1)
                else:
                    processed_space[param_name] = UniformDistribution(low, high)
            else:
                raise ValueError(f"Invalid tuple format for {param_name}: {param_spec}")
        elif isinstance(param_spec, list):
            # List format - categorical choice
            processed_space[param_name] = ChoiceDistribution(param_spec)
        else:
            raise ValueError(
                f"Unsupported search space format for {param_name}: {param_spec}"
            )

    optimizer = RandomSearchOptimizer(
        objective_function=objective_function,
        search_space=processed_space,
        n_iter=n_iter,
        random_state=random_state,
        n_jobs=n_jobs,
        early_stopping=early_stopping,
        patience=patience,
    )

    return optimizer.optimize(verbose=verbose)


def analyze_parameter_importance(
    result: RandomSearchResult, top_n: int = 10
) -> Dict[str, float]:
    """
    Analyze parameter importance using correlation with objective values.

    Parameters
    ----------
    result : RandomSearchResult
        Results from random search optimization
    top_n : int, default=10
        Number of top configurations to analyze

    Returns
    -------
    dict
        Parameter importance scores (correlation coefficients)
    """
    if len(result.all_params) < 2:
        return {}

    sorted_indices = np.argsort(result.all_scores)[-top_n:]
    top_params = [result.all_params[i] for i in sorted_indices]
    top_scores = [result.all_scores[i] for i in sorted_indices]

    importance_scores = {}
    param_names = list(result.all_params[0].keys())

    for param_name in param_names:
        param_values = []
        for params in top_params:
            val = params[param_name]
            if isinstance(val, (int, float)):
                param_values.append(float(val))
            else:
                # For categorical parameters, skip importance analysis
                continue

        if len(param_values) > 1:
            correlation = np.corrcoef(param_values, top_scores)[0, 1]
            if not np.isnan(correlation):
                importance_scores[param_name] = abs(correlation)

    importance_scores = dict(
        sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    )

    return importance_scores


def quadratic_objective(params):
    """Example objective - quadratic function with noise."""
    x, y = params["x"], params["y"]
    return -((x - 2) ** 2) - (y + 1) ** 2 - 5 + np.random.normal(0, 0.1)


def nn_objective(params):
    """Realistic neural network objective function simulation."""
    lr = params["learning_rate"]
    wd = params["weight_decay"]
    batch_size = params["batch_size"]
    dropout = params["dropout_rate"]
    optimizer = params["optimizer"]

    # Simulate realistic hyperparameter interactions
    base_score = 0.85

    lr_effect = -2 * (np.log10(lr) + 3) ** 2  # Optimum at 1e-3

    wd_effect = -0.5 * (np.log10(wd) + 4) ** 2  # Optimum around 1e-4

    batch_effect = -0.01 * (batch_size - 64) ** 2 / 100

    dropout_effect = -2 * (dropout - 0.2) ** 2

    optimizer_effects = {"adam": 0.05, "sgd": 0.0, "rmsprop": 0.02}
    opt_effect = optimizer_effects.get(optimizer, 0.0)

    score = (
        base_score + lr_effect + wd_effect + batch_effect + dropout_effect + opt_effect
    )
    score += np.random.normal(0, 0.02)  # Add realistic noise

    return score


def simple_nn_objective(params):
    """Simple neural network objective for testing."""
    lr = params["learning_rate"]
    wd = params["weight_decay"]
    return (
        -((np.log10(lr) + 3) ** 2) - (np.log10(wd) + 4) ** 2 + np.random.normal(0, 0.1)
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    print("Example 1: Quadratic Function Optimization")
    print("True optimum: x=2, y=-1, value=-5")

    search_space_simple = {"x": (-5, 5), "y": (-5, 5)}

    result1 = random_search(
        objective_function=quadratic_objective,
        search_space=search_space_simple,  # type: ignore
        n_iter=50,
        random_state=42,
        n_jobs=1,  # Use single job for this example
        verbose=True,
    )

    print(f"Found optimum: {result1.best_params}")
    print(f"Best value: {result1.best_score:.3f}")

    print("\n" + "=" * 60)
    print("Example 2: Neural Network Hyperparameter Optimization")

    # Define advanced search space
    search_space_advanced = {
        "learning_rate": LogUniformDistribution(1e-5, 1e-1),
        "weight_decay": LogUniformDistribution(1e-6, 1e-2),
        "batch_size": ChoiceDistribution(
            [16, 32, 64, 128, 256], probabilities=[0.1, 0.2, 0.4, 0.2, 0.1]
        ),
        "dropout_rate": UniformDistribution(0.0, 0.5),
        "optimizer": ChoiceDistribution(["adam", "sgd", "rmsprop"]),
    }

    result2 = random_search(
        objective_function=nn_objective,
        search_space=search_space_advanced,
        n_iter=100,
        n_jobs=4,
        early_stopping=True,
        patience=15,
        random_state=42,
    )

    print(f"\nBest hyperparameters:")
    if result2.best_params is not None:
        for param, value in result2.best_params.items():
            if isinstance(value, float):
                print(f"  {param}: {value:.6f}")
            else:
                print(f"  {param}: {value}")

        print(f"Best validation accuracy: {result2.best_score:.4f}")
    else:
        print("  No valid results found!")

    print(f"Search completed in {result2.search_time:.2f} seconds")

    if result2.best_params is not None:
        importance = analyze_parameter_importance(result2)
        print(f"\nParameter importance (top configurations):")
        for param, score in importance.items():
            print(f"  {param}: {score:.3f}")

    print("\n" + "=" * 60)
    print("Example 3: Simple Format Usage")

    search_space_simple_nn = {
        "learning_rate": (1e-5, 1e-1),
        "weight_decay": (1e-6, 1e-2),
        "batch_size": [16, 32, 64, 128],
        "hidden_units": (64, 512),
        "activation": ["relu", "tanh", "sigmoid"],
    }

    result3 = random_search(
        objective_function=simple_nn_objective,
        search_space=search_space_simple_nn,
        n_iter=30,
        random_state=42,
        n_jobs=2,
        verbose=False,  # Quiet mode
    )

    print(f"Simple format result:")
    print(f"  Best score: {result3.best_score:.4f}")
    if result3.best_params is not None:
        print(f"  Learning rate: {result3.best_params['learning_rate']:.2e}")
        print(f"  Weight decay: {result3.best_params['weight_decay']:.2e}")
        print(f"  Batch size: {result3.best_params['batch_size']}")
        print(f"  Hidden units: {result3.best_params['hidden_units']}")
        print(f"  Activation: {result3.best_params['activation']}")
    else:
        print("  No valid results found!")
