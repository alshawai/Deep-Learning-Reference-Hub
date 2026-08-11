"""
Modern Hyperparameter Tuning Framework
======================================

Production-ready framework integrating multiple optimization strategies with
experiment tracking and statistical analysis. This framework provides a unified
interface for various hyperparameter optimization methods and includes tools
for result comparison, visualization, and reproducibility.

References
----------
- Feurer, M., & Hutter, F. (2019). "Hyperparameter Optimization."
  Automated Machine Learning: Methods, Systems, Challenges.
- Liaw, R., et al. (2018). "Tune: A Research Platform for Distributed
  Model Selection and Training." arXiv preprint arXiv:1807.05118.

Author
------
Deep Learning Reference Hub

License
-------
MIT License

Notes
-----
This framework is designed to be:
1. Flexible - supports multiple optimization strategies
2. Extensible - easy to add new optimizers
3. Reproducible - proper random seeding and logging
4. Production-ready - includes error handling and checkpointing
"""

import json
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class OptimizationMethod(Enum):
    """Enumeration of available optimization methods."""

    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    ASHA = "asha"
    PBT = "pbt"
    GRID_SEARCH = "grid_search"


@dataclass
class HyperparameterConfig:
    """
    Configuration for a single hyperparameter.

    Attributes
    ----------
    name : str
        Parameter name
    type : str
        Parameter type ('continuous', 'integer', 'categorical')
    range : tuple or list
        Valid range or choices for the parameter
    scale : str, default='linear'
        Scale for sampling ('linear', 'log')
    default : Any (optional)
        Default value for said parameter
    """

    name: str
    type: str
    range: tuple | list
    scale: str = "linear"
    default: Any = None


@dataclass
class ExperimentConfig:
    """
    Configuration for hyperparameter optimization experiment.

    Attributes
    ----------
    experiment_name : str
        Name of the experiment
    optimization_method : OptimizationMethod
        Optimization strategy to use
    hyperparameters : list
        List of HyperparameterConfig objects
    objective_metric : str
        Name of metric to optimize
    maximize : bool, default=True
        Whether to maximize the objective metric
    n_trials : int, default=100
        Number of trials to run
    random_seed : int (optional)
        Random seed for reproducibility
    save_dir : str (optional)
        Directory to save results
    additional_config : dict (optional)
        Additional method-specific configuration
    """

    experiment_name: str
    optimization_method: OptimizationMethod
    hyperparameters: list[HyperparameterConfig]
    objective_metric: str
    maximize: bool = True
    n_trials: int = 100
    random_seed: int | None = None
    save_dir: str | None = None
    additional_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialResult:
    """
    Result from a single trial.

    Attributes
    ----------
    trial_id : int
        Unique trial identifier
    hyperparams : dict
        Hyperparameter configuration used
    metrics : dict
        All metrics recorded
    training_time : float
        Time taken for training
    status : str
        Trial status ('success', 'failed', 'cancelled')
    metadata : dict
        Additional trial information
    """

    trial_id: int
    hyperparams: dict[str, Any]
    metrics: dict[str, float]
    training_time: float
    status: str = "success"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """
    The outcome of one hyperparameter search: every trial, and the best of them.

    Named for the :class:`ExperimentConfig` it answers. Not to be confused with
    :class:`dlhub.optimizers.OptimizationRun`, which traces a single descent.

    Attributes
    ----------
    experiment_config : ExperimentConfig
        Configuration used for the experiment
    best_trial : TrialResult
        Best performing trial
    all_trials : list
        All trial results
    total_time : float
        Total optimization time
    summary_statistics : dict
        Summary statistics and analysis
    """

    experiment_config: ExperimentConfig
    best_trial: TrialResult
    all_trials: list[TrialResult]
    total_time: float
    summary_statistics: dict[str, Any] = field(default_factory=dict)


class ObjectiveFunction(ABC):
    """
    Abstract base class for objective functions.

    Defines the interface that objective functions must implement.
    """

    @abstractmethod
    def evaluate(self, hyperparams: dict[str, Any]) -> dict[str, float]:
        """
        Evaluate hyperparameters and return metrics.

        Parameters
        ----------
        hyperparams : dict
            Hyperparameter configuration

        Returns
        -------
        dict
            Dictionary of metric names to values
        """
        pass

    @abstractmethod
    def get_metric_names(self) -> list[str]:
        """
        Get names of all metrics returned by evaluate.

        Returns
        -------
        list
            List of metric names
        """
        pass


class FunctionObjective(ObjectiveFunction):
    """
    Wrapper for function-based objectives.

    Parameters
    ----------
    eval_function : callable
        Function that takes hyperparams and returns metrics dict
    metric_names : list
        Names of metrics returned by eval_function
    """

    def __init__(
        self, eval_function: Callable[[dict], dict[str, float]], metric_names: list[str]
    ):
        self.eval_function = eval_function
        self.metric_names = metric_names

    def evaluate(self, hyperparams: dict[str, Any]) -> dict[str, float]:
        """Evaluate using wrapped function."""
        return self.eval_function(hyperparams)

    def get_metric_names(self) -> list[str]:
        """Get metric names."""
        return self.metric_names


class HyperparameterSampler:
    """
    Utility class for sampling hyperparameters from configurations.

    Parameters
    ----------
    hyperparameter_configs : list
        List of HyperparameterConfig objects
    random_state : int, optional
        Random seed for reproducibility
    """

    def __init__(
        self,
        hyperparameter_configs: list[HyperparameterConfig],
        random_state: int | None = None,
    ):
        self.configs = hyperparameter_configs
        if random_state is not None:
            np.random.seed(random_state)

    def sample(self) -> dict[str, Any]:
        """
        Sample a hyperparameter configuration.

        Returns
        -------
        dict
            Sampled hyperparameter configuration
        """
        hyperparams = {}

        for config in self.configs:
            if config.type == "continuous":
                hyperparams[config.name] = self._sample_continuous(config)
            elif config.type == "integer":
                hyperparams[config.name] = self._sample_integer(config)
            elif config.type == "categorical":
                hyperparams[config.name] = self._sample_categorical(config)
            else:
                raise ValueError(f"Unknown parameter type: {config.type}")

        return hyperparams

    def _sample_continuous(self, config: HyperparameterConfig) -> float:
        """Sample continuous parameter."""
        low, high = config.range

        if config.scale == "log":
            return np.exp(np.random.uniform(np.log(low), np.log(high)))
        else:
            return np.random.uniform(low, high)

    def _sample_integer(self, config: HyperparameterConfig) -> int:
        """Sample integer parameter."""
        low, high = config.range

        if config.scale == "log":
            log_value = np.random.uniform(np.log(low), np.log(high))
            return int(np.round(np.exp(log_value)))
        else:
            return np.random.randint(low, high + 1)

    def _sample_categorical(self, config: HyperparameterConfig) -> Any:
        """Sample categorical parameter."""
        return np.random.choice(config.range)

    def validate(self, hyperparams: dict[str, Any]) -> bool:
        """
        Validate a hyperparameter configuration.

        Parameters
        ----------
        hyperparams : dict
            Hyperparameter configuration to validate

        Returns
        -------
        bool
            True if configuration is valid
        """
        for config in self.configs:
            if config.name not in hyperparams:
                return False

            value = hyperparams[config.name]

            if config.type in ["continuous", "integer"]:
                low, high = config.range
                if not (low <= value <= high):
                    return False
            elif config.type == "categorical":
                if value not in config.range:
                    return False

        return True


class ExperimentLogger:
    """
    Logger for experiment results and metadata.

    Parameters
    ----------
    save_dir : str (optional)
        Directory to save logs
    experiment_name : str
        Name of the experiment
    """

    def __init__(self, save_dir: str | None, experiment_name: str):
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir) if save_dir else None

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = self.save_dir / f"{experiment_name}_log.jsonl"
        else:
            self.log_file = None

    def log_trial(self, trial_result: TrialResult) -> None:
        """
        Log a trial result.

        Parameters
        ----------
        trial_result : TrialResult
            Trial result to log
        """
        if self.log_file is None:
            return

        # Convert to dictionary
        trial_dict = {
            "trial_id": trial_result.trial_id,
            "hyperparams": trial_result.hyperparams,
            "metrics": trial_result.metrics,
            "training_time": trial_result.training_time,
            "status": trial_result.status,
            "metadata": trial_result.metadata,
            "timestamp": time.time(),
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(trial_dict) + "\n")

    def save_results(self, result: ExperimentResult) -> None:
        """
        Save complete optimization results.

        Parameters
        ----------
        result : ExperimentResult
            Optimization results to save
        """
        if self.save_dir is None:
            return

        summary_file = self.save_dir / f"{self.experiment_name}_summary.json"
        summary = {
            "experiment_name": result.experiment_config.experiment_name,
            "optimization_method": result.experiment_config.optimization_method.value,
            "best_trial": {
                "hyperparams": result.best_trial.hyperparams,
                "metrics": result.best_trial.metrics,
                "trial_id": result.best_trial.trial_id,
            },
            "total_time": result.total_time,
            "n_trials": len(result.all_trials),
            "summary_statistics": result.summary_statistics,
        }

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

    def load_results(self) -> list[TrialResult] | None:
        """
        Load trial results from log file.

        Returns
        -------
        list or None
            List of TrialResult objects, or None if no log exists
        """
        if self.log_file is None or not self.log_file.exists():
            return None

        trials = []
        with open(self.log_file) as f:
            for line in f:
                trial_dict = json.loads(line)
                trial = TrialResult(
                    trial_id=trial_dict["trial_id"],
                    hyperparams=trial_dict["hyperparams"],
                    metrics=trial_dict["metrics"],
                    training_time=trial_dict["training_time"],
                    status=trial_dict["status"],
                    metadata=trial_dict["metadata"],
                )
                trials.append(trial)

        return trials


class HyperparameterOptimizer:
    """
    Main hyperparameter optimization framework.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration
    objective : ObjectiveFunction
        Objective function to optimize
    """

    def __init__(self, config: ExperimentConfig, objective: ObjectiveFunction):
        self.config = config
        self.objective = objective

        if config.random_seed is not None:
            np.random.seed(config.random_seed)

        self.sampler = HyperparameterSampler(config.hyperparameters, config.random_seed)

        self.logger = ExperimentLogger(config.save_dir, config.experiment_name)

        self.all_trials = []
        self.best_trial = None
        self.trial_counter = 0

    def _evaluate_trial(self, hyperparams: dict[str, Any]) -> TrialResult:
        """
        Evaluate a single trial.

        Parameters
        ----------
        hyperparams : dict
            Hyperparameter configuration

        Returns
        -------
        TrialResult
            Result of the trial
        """
        trial_id = self.trial_counter
        self.trial_counter += 1

        start_time = time.time()

        try:
            if not self.sampler.validate(hyperparams):
                raise ValueError("Invalid hyperparameter configuration")

            metrics = self.objective.evaluate(hyperparams)
            training_time = time.time() - start_time

            if self.config.objective_metric not in metrics:
                raise ValueError(
                    f"Objective metric '{self.config.objective_metric}' "
                    "not found in results"
                )

            result = TrialResult(
                trial_id=trial_id,
                hyperparams=hyperparams.copy(),
                metrics=metrics,
                training_time=training_time,
                status="success",
            )

        except Exception as e:
            training_time = time.time() - start_time
            warnings.warn(f"Trial {trial_id} failed: {e}")

            result = TrialResult(
                trial_id=trial_id,
                hyperparams=hyperparams.copy(),
                metrics={
                    self.config.objective_metric: -np.inf
                    if self.config.maximize
                    else np.inf
                },
                training_time=training_time,
                status="failed",
                metadata={"error": str(e)},
            )

        self.logger.log_trial(result)
        self.all_trials.append(result)

        self._update_best_trial(result)

        return result

    def _update_best_trial(self, trial: TrialResult) -> None:
        """Update best trial if current trial is better."""
        if trial.status != "success":
            return

        current_score = trial.metrics[self.config.objective_metric]

        if self.best_trial is None:
            self.best_trial = trial
        else:
            best_score = self.best_trial.metrics[self.config.objective_metric]

            if self.config.maximize:
                if current_score > best_score:
                    self.best_trial = trial
            else:
                if current_score < best_score:
                    self.best_trial = trial

    def optimize(self, verbose: bool = True) -> ExperimentResult:
        """
        Run hyperparameter optimization.

        Parameters
        ----------
        verbose : bool, default=True
            Whether to print progress

        Returns
        -------
        ExperimentResult
            Optimization results
        """
        if verbose:
            print(
                f"Starting Hyperparameter Optimization: {self.config.experiment_name}"
            )
            print(f"Method: {self.config.optimization_method.value}")
            print(f"Number of trials: {self.config.n_trials}")
            print(
                f"Objective: {'maximize' if self.config.maximize else 'minimize'} "
                f"{self.config.objective_metric}"
            )

        start_time = time.time()

        if self.config.optimization_method == OptimizationMethod.RANDOM_SEARCH:
            self._run_random_search(verbose)
        elif self.config.optimization_method == OptimizationMethod.GRID_SEARCH:
            self._run_grid_search(verbose)
        else:
            raise NotImplementedError(
                f"Method {self.config.optimization_method.value} not implemented "
                "in this simplified framework"
            )

        total_time = time.time() - start_time

        summary_stats = self._compute_summary_statistics()

        result = ExperimentResult(
            experiment_config=self.config,
            best_trial=self.best_trial,  # type: ignore
            all_trials=self.all_trials,
            total_time=total_time,
            summary_statistics=summary_stats,
        )

        self.logger.save_results(result)

        if verbose:
            print(f"\nOptimization completed in {total_time:.2f} seconds")
            print(
                f"Best {self.config.objective_metric}: "
                f"{self.best_trial.metrics[self.config.objective_metric]:.6f}"
            )  # type: ignore
            print(f"Best hyperparameters: {self.best_trial.hyperparams}")  # type: ignore

        return result

    def _run_random_search(self, verbose: bool) -> None:
        """Run random search optimization."""
        for i in range(self.config.n_trials):
            hyperparams = self.sampler.sample()
            trial = self._evaluate_trial(hyperparams)

            if verbose and (i + 1) % max(1, self.config.n_trials // 10) == 0:
                print(
                    f"Trial {i + 1}/{self.config.n_trials}: "
                    f"{self.config.objective_metric} = "
                    f"{trial.metrics.get(self.config.objective_metric, 'N/A')}"
                )

    def _run_grid_search(self, verbose: bool) -> None:
        """Run grid search optimization."""
        grid_configs = self._generate_grid()

        if verbose:
            print(f"Generated grid with {len(grid_configs)} configurations")

        for i, hyperparams in enumerate(grid_configs[: self.config.n_trials]):
            trial = self._evaluate_trial(hyperparams)

            if verbose and (i + 1) % max(1, len(grid_configs) // 10) == 0:
                print(
                    f"Trial {i + 1}/{min(len(grid_configs), self.config.n_trials)}: "
                    f"{self.config.objective_metric} = "
                    f"{trial.metrics.get(self.config.objective_metric, 'N/A')}"
                )

    def _generate_grid(self) -> list[dict[str, Any]]:
        """Generate grid of hyperparameter configurations."""
        from itertools import product

        param_grids = {}
        for config in self.config.hyperparameters:
            if config.type == "categorical":
                param_grids[config.name] = config.range
            elif config.type in ["continuous", "integer"]:
                n_points = 5  # Use 5 points per dimension
                low, high = config.range

                if config.scale == "log":
                    points = np.logspace(np.log10(low), np.log10(high), n_points)
                else:
                    points = np.linspace(low, high, n_points)

                if config.type == "integer":
                    points = np.unique(np.round(points).astype(int))

                param_grids[config.name] = points.tolist()

        keys = list(param_grids.keys())
        values = list(param_grids.values())

        grid_configs = []
        for combo in product(*values):
            config_dict = dict(zip(keys, combo))
            grid_configs.append(config_dict)

        return grid_configs

    def _compute_summary_statistics(self) -> dict[str, Any]:
        """Compute summary statistics from all trials."""
        successful_trials = [t for t in self.all_trials if t.status == "success"]

        if not successful_trials:
            return {}

        objective_scores = [
            t.metrics[self.config.objective_metric] for t in successful_trials
        ]

        training_times = [t.training_time for t in successful_trials]

        statistics = {
            "n_trials": len(self.all_trials),
            "n_successful": len(successful_trials),
            "n_failed": len(self.all_trials) - len(successful_trials),
            "objective_statistics": {
                "mean": float(np.mean(objective_scores)),
                "std": float(np.std(objective_scores)),
                "min": float(np.min(objective_scores)),
                "max": float(np.max(objective_scores)),
                "median": float(np.median(objective_scores)),
                "percentiles": {
                    "25th": float(np.percentile(objective_scores, 25)),
                    "75th": float(np.percentile(objective_scores, 75)),
                    "95th": float(np.percentile(objective_scores, 95)),
                },
            },
            "training_time_statistics": {
                "mean": float(np.mean(training_times)),
                "total": float(np.sum(training_times)),
                "min": float(np.min(training_times)),
                "max": float(np.max(training_times)),
            },
            "improvement_over_random": self._improvement_over_random(objective_scores),
        }

        return statistics

    def _improvement_over_random(self, objective_scores: list[float]) -> float:
        """
        Relative gain of the best trial over the first ten, which for random and
        grid search are as good a stand-in for chance as the run provides.

        Reported as 0.0 below ten trials, and when the baseline mean is zero:
        the relative gain over nothing is not a number, and a metric centred on
        zero is a legitimate thing to optimize.
        """
        if len(objective_scores) < 10:
            return 0.0

        baseline = float(np.mean(objective_scores[:10]))
        if baseline == 0.0:
            return 0.0

        best_score = self.best_trial.metrics[self.config.objective_metric]  # type: ignore
        return float((best_score - baseline) / abs(baseline))


def optimize_hyperparameters(
    objective_function: Callable[[dict], dict[str, float]],
    hyperparameters: list[dict[str, Any]],
    objective_metric: str,
    experiment_name: str = "hyperparameter_optimization",
    optimization_method: str = "random_search",
    n_trials: int = 100,
    maximize: bool = True,
    random_seed: int | None = None,
    save_dir: str | None = None,
    verbose: bool = True,
) -> ExperimentResult:
    """
    Run a hyperparameter search by method name, returning an ``ExperimentResult``.

    The general dispatcher: ``optimization_method`` selects the strategy. It takes
    the framework's search-space format, which carries a type per hyperparameter
    and so covers categorical and integer dimensions as well as continuous ones.
    The method-specific entry points -- :func:`~dlhub.tuning.bayesian_optimize`,
    :func:`~dlhub.tuning.random_search` -- take their own formats and return their
    own result types.

    Parameters
    ----------
    objective_function : callable
        Function that takes hyperparams dict and returns metrics dict
    hyperparameters : list
        List of hyperparameter definitions (dicts with 'name', 'type', 'range')
    objective_metric : str
        Name of metric to optimize
    experiment_name : str, default="hyperparameter_optimization"
        Name of the experiment
    optimization_method : str, default="random_search"
        Optimization method ('random_search' or 'grid_search')
    n_trials : int, default=100
        Number of trials
    maximize : bool, default=True
        Whether to maximize objective
    random_seed : int (optional)
        Random seed
    save_dir : str (optional)
        Directory to save results
    verbose : bool, default=True
        Whether to print progress

    Returns
    -------
    ExperimentResult
        Optimization results

    Examples
    --------
    >>> def objective(hyperparams):
    ...     lr = hyperparams["learning_rate"]
    ...     wd = hyperparams["weight_decay"]
    ...     accuracy = 0.9 - (lr - 0.001) ** 2 - (wd - 0.0001) ** 2
    ...     return {"accuracy": accuracy, "loss": 1 - accuracy}
    >>> hyperparams = [
    ...     {
    ...         "name": "learning_rate",
    ...         "type": "continuous",
    ...         "range": (1e-5, 1e-1),
    ...         "scale": "log",
    ...     },
    ...     {
    ...         "name": "weight_decay",
    ...         "type": "continuous",
    ...         "range": (1e-6, 1e-2),
    ...         "scale": "log",
    ...     },
    ... ]
    >>> result = optimize_hyperparameters(
    ...     objective, hyperparams, "accuracy", n_trials=50, random_seed=42
    ... )
    """
    hp_configs = []
    for hp in hyperparameters:
        config = HyperparameterConfig(
            name=hp["name"],
            type=hp["type"],
            range=hp["range"],
            scale=hp.get("scale", "linear"),
            default=hp.get("default"),
        )
        hp_configs.append(config)

    exp_config = ExperimentConfig(
        experiment_name=experiment_name,
        optimization_method=OptimizationMethod(optimization_method),
        hyperparameters=hp_configs,
        objective_metric=objective_metric,
        maximize=maximize,
        n_trials=n_trials,
        random_seed=random_seed,
        save_dir=save_dir,
    )

    test_hyperparams = HyperparameterSampler(hp_configs, random_seed).sample()
    test_metrics = objective_function(test_hyperparams)
    metric_names = list(test_metrics.keys())

    objective_wrapper = FunctionObjective(objective_function, metric_names)

    optimizer = HyperparameterOptimizer(exp_config, objective_wrapper)
    return optimizer.optimize(verbose=verbose)


if __name__ == "__main__":
    print("Modern Hyperparameter Tuning Framework Example")
    print("=" * 60)

    def nn_objective(hyperparams: dict[str, Any]) -> dict[str, float]:
        """
        Simulate neural network training objective.

        Returns multiple metrics for comprehensive evaluation.
        """
        lr = hyperparams["learning_rate"]
        wd = hyperparams["weight_decay"]
        batch_size = hyperparams["batch_size"]
        dropout = hyperparams["dropout_rate"]
        optimizer_type = hyperparams["optimizer"]

        # Simulate realistic hyperparameter effects
        # Optimal values: lr=1e-3, wd=1e-4, batch_size=64, dropout=0.2
        lr_effect = -5 * (np.log10(lr) + 3) ** 2
        wd_effect = -2 * (np.log10(wd) + 4) ** 2
        batch_effect = -0.001 * (batch_size - 64) ** 2
        dropout_effect = -3 * (dropout - 0.2) ** 2

        optimizer_effects = {"adam": 0.05, "sgd": 0.0, "rmsprop": 0.03}
        opt_effect = optimizer_effects.get(optimizer_type, 0.0)

        base_accuracy = 0.85

        validation_accuracy = (
            base_accuracy
            + lr_effect
            + wd_effect
            + batch_effect
            + dropout_effect
            + opt_effect
        )

        validation_accuracy += np.random.normal(0, 0.02)

        training_accuracy = validation_accuracy + 0.05  # Training usually higher
        training_loss = 1.0 - training_accuracy
        validation_loss = 1.0 - validation_accuracy

        inference_time = 10.0 / batch_size + dropout * 5.0
        model_size = 100.0 * (1 - dropout * 0.3)

        return {
            "validation_accuracy": validation_accuracy,
            "training_accuracy": training_accuracy,
            "validation_loss": validation_loss,
            "training_loss": training_loss,
            "inference_time_ms": inference_time,
            "model_size_mb": model_size,
        }

    hyperparameter_space = [
        {
            "name": "learning_rate",
            "type": "continuous",
            "range": (1e-5, 1e-1),
            "scale": "log",
            "default": 1e-3,
        },
        {
            "name": "weight_decay",
            "type": "continuous",
            "range": (1e-6, 1e-2),
            "scale": "log",
            "default": 1e-4,
        },
        {
            "name": "batch_size",
            "type": "integer",
            "range": (16, 256),
            "scale": "linear",
        },
        {
            "name": "dropout_rate",
            "type": "continuous",
            "range": (0.0, 0.5),
            "scale": "linear",
            "default": 0.1,
        },
        {
            "name": "optimizer",
            "type": "categorical",
            "range": ["adam", "sgd", "rmsprop"],
            "default": "adam",
        },
    ]

    print("\nRunning Random Search optimization...")
    result = optimize_hyperparameters(
        objective_function=nn_objective,
        hyperparameters=hyperparameter_space,
        objective_metric="validation_accuracy",
        experiment_name="neural_network_tuning",
        optimization_method="random_search",
        n_trials=50,
        maximize=True,
        random_seed=42,
        save_dir="./hyperparameter_results",
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)

    print(f"\nBest Configuration:")
    for param, value in result.best_trial.hyperparams.items():
        if isinstance(value, float):
            print(f"  {param}: {value:.6f}")
        else:
            print(f"  {param}: {value}")

    print(f"\nBest Performance Metrics:")
    for metric, value in result.best_trial.metrics.items():
        print(f"  {metric}: {value:.6f}")

    print(f"\nSummary Statistics:")
    print(f"  Total trials: {result.summary_statistics['n_trials']}")
    print(f"  Successful trials: {result.summary_statistics['n_successful']}")
    print(f"  Failed trials: {result.summary_statistics['n_failed']}")
    print(f"  Total time: {result.total_time:.2f} seconds")

    obj_stats = result.summary_statistics["objective_statistics"]
    print(f"\n{result.experiment_config.objective_metric.capitalize()} Statistics:")
    print(f"  Mean: {obj_stats['mean']:.6f}")
    print(f"  Std: {obj_stats['std']:.6f}")
    print(f"  Min: {obj_stats['min']:.6f}")
    print(f"  Max: {obj_stats['max']:.6f}")
    print(f"  Median: {obj_stats['median']:.6f}")

    print(f"\nPercentiles:")
    for percentile, value in obj_stats["percentiles"].items():
        print(f"  {percentile}: {value:.6f}")

    time_stats = result.summary_statistics["training_time_statistics"]
    print(f"\nTraining Time Statistics:")
    print(f"  Mean per trial: {time_stats['mean']:.4f} seconds")
    print(f"  Total time: {time_stats['total']:.2f} seconds")
    print(f"  Min: {time_stats['min']:.4f} seconds")
    print(f"  Max: {time_stats['max']:.4f} seconds")

    improvement = result.summary_statistics["improvement_over_random"]
    print(f"\nImprovement over random baseline: {improvement:.2%}")

    print("\n" + "=" * 60)
    print("TOP 5 CONFIGURATIONS")
    print("=" * 60)

    successful_trials = [t for t in result.all_trials if t.status == "success"]
    sorted_trials = sorted(
        successful_trials,
        key=lambda t: t.metrics[result.experiment_config.objective_metric],
        reverse=result.experiment_config.maximize,
    )

    for i, trial in enumerate(sorted_trials[:5], 1):
        print(f"\n#{i} - Trial {trial.trial_id}")
        print(
            f"  {result.experiment_config.objective_metric}: "
            f"{trial.metrics[result.experiment_config.objective_metric]:.6f}"
        )
        print(f"  Hyperparameters:")
        for param, value in trial.hyperparams.items():
            if isinstance(value, float):
                print(f"    {param}: {value:.6f}")
            else:
                print(f"    {param}: {value}")

    print("\n" + "=" * 60)
    print("GRID SEARCH COMPARISON")
    print("=" * 60)

    grid_hyperparameters = [
        {
            "name": "learning_rate",
            "type": "continuous",
            "range": (1e-4, 1e-2),
            "scale": "log",
        },
        {
            "name": "weight_decay",
            "type": "continuous",
            "range": (1e-5, 1e-3),
            "scale": "log",
        },
        {"name": "batch_size", "type": "categorical", "range": [32, 64, 128]},
        {
            "name": "dropout_rate",
            "type": "continuous",
            "range": (0.1, 0.3),
            "scale": "linear",
        },
        {"name": "optimizer", "type": "categorical", "range": ["adam", "rmsprop"]},
    ]

    print("\nRunning Grid Search (smaller space)...")
    grid_result = optimize_hyperparameters(
        objective_function=nn_objective,
        hyperparameters=grid_hyperparameters,
        objective_metric="validation_accuracy",
        experiment_name="neural_network_grid_search",
        optimization_method="grid_search",
        n_trials=50,
        maximize=True,
        random_seed=42,
        verbose=False,
    )

    print(f"\nGrid Search Results:")
    print(
        f"  Best validation accuracy: "
        f"{grid_result.best_trial.metrics['validation_accuracy']:.6f}"
    )
    print(f"  Total trials: {len(grid_result.all_trials)}")
    print(f"  Total time: {grid_result.total_time:.2f} seconds")

    print("\n" + "=" * 60)
    print("METHOD COMPARISON")
    print("=" * 60)

    random_best = result.best_trial.metrics["validation_accuracy"]
    grid_best = grid_result.best_trial.metrics["validation_accuracy"]

    print(f"\nRandom Search:")
    print(f"  Best accuracy: {random_best:.6f}")
    print(f"  Trials: {len(result.all_trials)}")
    print(f"  Time: {result.total_time:.2f}s")
    print(f"  Efficiency: {random_best / result.total_time:.6f} acc/sec")

    print(f"\nGrid Search:")
    print(f"  Best accuracy: {grid_best:.6f}")
    print(f"  Trials: {len(grid_result.all_trials)}")
    print(f"  Time: {grid_result.total_time:.2f}s")
    print(f"  Efficiency: {grid_best / grid_result.total_time:.6f} acc/sec")

    if random_best > grid_best:
        print(f"\nRandom Search found better solution by {random_best - grid_best:.6f}")
    else:
        print(f"\nGrid Search found better solution by {grid_best - random_best:.6f}")

    print("\n" + "=" * 60)
    print("MULTI-OBJECTIVE ANALYSIS")
    print("=" * 60)

    print("\nPareto-optimal configurations (accuracy vs. inference time):")

    # Find Pareto-optimal solutions
    def is_dominated(trial1, trial2):
        """Check if trial1 is dominated by trial2."""
        acc1 = trial1.metrics["validation_accuracy"]
        time1 = trial1.metrics["inference_time_ms"]
        acc2 = trial2.metrics["validation_accuracy"]
        time2 = trial2.metrics["inference_time_ms"]

        # trial1 is dominated if trial2 is better in both objectives
        return acc2 >= acc1 and time2 <= time1 and (acc2 > acc1 or time2 < time1)

    pareto_trials = []
    for trial in successful_trials:
        dominated = False
        for other in successful_trials:
            if trial.trial_id != other.trial_id and is_dominated(trial, other):
                dominated = True
                break
        if not dominated:
            pareto_trials.append(trial)

    # Sort by accuracy
    pareto_trials.sort(key=lambda t: t.metrics["validation_accuracy"], reverse=True)

    print(f"\nFound {len(pareto_trials)} Pareto-optimal configurations:")
    for i, trial in enumerate(pareto_trials[:5], 1):
        acc = trial.metrics["validation_accuracy"]
        time = trial.metrics["inference_time_ms"]
        size = trial.metrics["model_size_mb"]
        print(f"\n  #{i} - Trial {trial.trial_id}")
        print(f"    Accuracy: {acc:.6f}")
        print(f"    Inference time: {time:.2f} ms")
        print(f"    Model size: {size:.2f} MB")
        print(f"    Learning rate: {trial.hyperparams['learning_rate']:.2e}")
        print(f"    Batch size: {trial.hyperparams['batch_size']}")

    print("\n" + "=" * 60)
    print("Framework demonstration completed!")
    print(f"Results saved to: ./hyperparameter_results/")
    print("=" * 60)
