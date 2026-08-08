"""
Population-Based Training (PBT) for Hyperparameter Optimization
===============================================================

Complete PBT implementation with online hyperparameter adaptation during training.
PBT simultaneously trains multiple models with different hyperparameters and
periodically updates hyperparameters based on population performance, enabling
discovery of time-varying optimal hyperparameters.

References
----------
- Jaderberg, M., et al. (2017). "Population Based Training of Neural Networks."
  arXiv preprint arXiv:1711.09846.
- Parker-Holder, J., et al. (2020). "Effective Diversity in Population Based
  Reinforcement Learning." NeurIPS.

Author
------
Deep Learning Reference Hub

License
-------
MIT License

Notes
-----
PBT is particularly effective for:
1. Long training runs where optimal hyperparameters may change over time
2. Scenarios where early performance may not predict final performance
3. Reinforcement learning where environment complexity increases
4. Large-scale distributed training with multiple workers
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
import copy
import threading
import warnings


@dataclass
class WorkerState:
    """
    State of a single worker in the population.

    Attributes
    ----------
    worker_id : int
        Unique identifier for the worker
    hyperparams : dict
        Current hyperparameter configuration
    performance_history : list
        History of performance scores
    training_step : int
        Current training step
    model_state : any
        Current model state (implementation dependent)
    metadata : dict
        Additional worker metadata
    """

    worker_id: int
    hyperparams: Dict[str, Any]
    performance_history: List[float] = field(default_factory=list)
    training_step: int = 0
    model_state: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PBTResult:
    """
    Results from Population-Based Training.

    Attributes
    ----------
    best_worker : WorkerState
        Best performing worker at the end
    final_population : list
        Final state of all workers
    population_history : list
        History of population states over time
    total_training_time : float
        Total time spent training
    total_steps : int
        Total training steps across all workers
    statistics : dict
        Training statistics and analysis
    """

    best_worker: WorkerState
    final_population: List[WorkerState]
    population_history: List[List[WorkerState]]
    total_training_time: float
    total_steps: int
    statistics: Dict[str, Any] = field(default_factory=dict)


class HyperparameterDistribution(ABC):
    """Abstract class for hyperparameter distributions used in exploration."""

    @abstractmethod
    def perturb(self, value: Any) -> Any:
        """
        Perturb a hyperparameter value.

        Parameters
        ----------
        value : any
            Current hyperparameter value

        Returns
        -------
        any
            Perturbed hyperparameter value
        """
        pass

    @abstractmethod
    def resample(self) -> Any:
        """
        Resample a hyperparameter value from the distribution.

        Returns
        -------
        any
            New hyperparameter value
        """
        pass


class LogUniformPerturbation(HyperparameterDistribution):
    """
    Log-uniform perturbation for hyperparameters that vary over orders of magnitude.

    Parameters
    ----------
    factor_range : tuple, default=(0.8, 1.2)
        Range of multiplicative factors for perturbation
    bounds : tuple, optional
        (min, max) bounds for the hyperparameter
    """

    def __init__(
        self,
        factor_range: Tuple[float, float] = (0.8, 1.2),
        bounds: Optional[Tuple[float, float]] = None,
    ):
        self.factor_range = factor_range
        self.bounds = bounds

    def perturb(self, value: float) -> float:
        """Perturb value by random multiplicative factor."""
        factor = np.random.uniform(*self.factor_range)
        new_value = value * factor

        if self.bounds is not None:
            new_value = np.clip(new_value, *self.bounds)

        return new_value

    def resample(self) -> float:
        """Resample from log-uniform distribution."""
        if self.bounds is None:
            raise ValueError("Bounds required for resampling")
        return np.exp(np.random.uniform(np.log(self.bounds[0]), np.log(self.bounds[1])))


class UniformPerturbation(HyperparameterDistribution):
    """
    Uniform perturbation for continuous hyperparameters.

    Parameters
    ----------
    noise_std : float, default=0.1
        Standard deviation of Gaussian noise to add
    bounds : tuple, optional
        (min, max) bounds for the hyperparameter
    """

    def __init__(
        self, noise_std: float = 0.1, bounds: Optional[Tuple[float, float]] = None
    ):
        self.noise_std = noise_std
        self.bounds = bounds

    def perturb(self, value: float) -> float:
        """Perturb value by adding Gaussian noise."""
        new_value = value + np.random.normal(0, self.noise_std)

        if self.bounds is not None:
            new_value = np.clip(new_value, *self.bounds)

        return new_value

    def resample(self) -> float:
        """Resample from uniform distribution."""
        if self.bounds is None:
            raise ValueError("Bounds required for resampling")
        return np.random.uniform(*self.bounds)


class ChoicePerturbation(HyperparameterDistribution):
    """
    Perturbation for categorical hyperparameters.

    Parameters
    ----------
    choices : list
        List of possible values
    change_probability : float, default=0.3
        Probability of changing to a different value
    """

    def __init__(self, choices: List[Any], change_probability: float = 0.3):
        self.choices = choices
        self.change_probability = change_probability

    def perturb(self, value: Any) -> Any:
        """Perturb categorical value."""
        if np.random.random() < self.change_probability:
            # Choose different value
            other_choices = [c for c in self.choices if c != value]
            if other_choices:
                return np.random.choice(other_choices)
        return value

    def resample(self) -> Any:
        """Resample from choices."""
        return np.random.choice(self.choices)


class WorkerInterface(ABC):
    """
    Abstract interface for training workers in PBT.

    Defines the methods that workers must implement to participate
    in population-based training.
    """

    @abstractmethod
    def train_step(
        self, hyperparams: Dict[str, Any], steps: int = 1
    ) -> Tuple[float, Any]:
        """
        Train for specified number of steps.

        Parameters
        ----------
        hyperparams : dict
            Current hyperparameter configuration
        steps : int, default=1
            Number of training steps to perform

        Returns
        -------
        tuple
            (performance_score, model_state)
        """
        pass

    @abstractmethod
    def save_state(self) -> Any:
        """
        Save current model state.

        Returns
        -------
        any
            Serializable model state
        """
        pass

    @abstractmethod
    def load_state(self, state: Any) -> None:
        """
        Load model state.

        Parameters
        ----------
        state : any
            Model state to load
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset worker to initial state."""
        pass


class FunctionWorker(WorkerInterface):
    """
    Function-based worker implementation.

    Wraps user-provided training functions to conform to WorkerInterface.

    Parameters
    ----------
    train_function : callable
        Function that takes (hyperparams, steps) and returns (score, state)
    save_function : callable
        Function that returns current state
    load_function : callable
        Function that loads given state
    reset_function : callable
        Function that resets to initial state
    """

    def __init__(
        self,
        train_function: Callable,
        save_function: Callable,
        load_function: Callable,
        reset_function: Callable,
    ):
        self.train_function = train_function
        self.save_function = save_function
        self.load_function = load_function
        self.reset_function = reset_function

    def train_step(
        self, hyperparams: Dict[str, Any], steps: int = 1
    ) -> Tuple[float, Any]:
        """Train using wrapped function."""
        return self.train_function(hyperparams, steps)

    def save_state(self) -> Any:
        """Save state using wrapped function."""
        return self.save_function()

    def load_state(self, state: Any) -> None:
        """Load state using wrapped function."""
        self.load_function(state)

    def reset(self) -> None:
        """Reset using wrapped function."""
        self.reset_function()


class PopulationBasedTrainer:
    """
    Population-Based Training optimizer.

    Manages a population of workers, periodically evaluating performance
    and updating hyperparameters through exploitation and exploration.

    Parameters
    ----------
    worker_factory : callable
        Factory function that creates new WorkerInterface instances
    initial_hyperparams : list
        Initial hyperparameter configurations for population
    hyperparam_distributions : dict
        Mapping from hyperparameter names to HyperparameterDistribution objects
    population_size : int, default=10
        Size of the population
    eval_interval : int, default=100
        Training steps between population evaluations
    exploit_fraction : float, default=0.2
        Fraction of worst performers to replace
    explore_fraction : float, default=0.2
        Fraction of hyperparameters to perturb during exploration
    truncation_selection : bool, default=True
        Whether to use truncation selection (replace worst with best)
    random_state : int, optional
        Random seed for reproducibility
    """

    def __init__(
        self,
        worker_factory: Callable[[], WorkerInterface],
        initial_hyperparams: List[Dict[str, Any]],
        hyperparam_distributions: Dict[str, HyperparameterDistribution],
        population_size: int = 10,
        eval_interval: int = 100,
        exploit_fraction: float = 0.2,
        explore_fraction: float = 0.2,
        truncation_selection: bool = True,
        random_state: Optional[int] = None,
    ):

        self.worker_factory = worker_factory
        self.initial_hyperparams = initial_hyperparams
        self.hyperparam_distributions = hyperparam_distributions
        self.population_size = population_size
        self.eval_interval = eval_interval
        self.exploit_fraction = exploit_fraction
        self.explore_fraction = explore_fraction
        self.truncation_selection = truncation_selection

        if random_state is not None:
            np.random.seed(random_state)

        # Initialize population
        self.population = []
        self.population_history = []
        self.total_steps = 0
        self.generation = 0

    def _initialize_population(self) -> None:
        """Initialize the population of workers."""
        self.population = []

        configs = self.initial_hyperparams.copy()
        while len(configs) < self.population_size:
            config = {}
            for param, dist in self.hyperparam_distributions.items():
                config[param] = dist.resample()
            configs.append(config)

        for i in range(self.population_size):
            config = configs[i % len(configs)]
            worker_state = WorkerState(
                worker_id=i,
                hyperparams=config.copy(),
                performance_history=[],
                training_step=0,
                model_state=None,
                metadata={"generation_created": 0},
            )
            self.population.append(worker_state)

    def _evaluate_population(self, workers: List[WorkerInterface]) -> List[float]:
        """
        Evaluate current performance of all workers.

        Parameters
        ----------
        workers : list
            List of worker instances

        Returns
        -------
        list
            Performance scores for each worker
        """
        scores = []
        for i, worker in enumerate(workers):
            try:
                # Train for evaluation interval
                score, model_state = worker.train_step(
                    self.population[i].hyperparams, self.eval_interval
                )

                # Update worker state
                self.population[i].performance_history.append(score)
                self.population[i].training_step += self.eval_interval
                self.population[i].model_state = model_state

                scores.append(score)

            except Exception as e:
                warnings.warn(f"Worker {i} evaluation failed: {e}")
                scores.append(-np.inf)

        return scores

    def _exploit_and_explore(
        self, workers: List[WorkerInterface], scores: List[float]
    ) -> None:
        """
        Perform exploitation and exploration step.

        Parameters
        ----------
        workers : list
            List of worker instances
        scores : list
            Current performance scores
        """
        if len(scores) < 2:
            return

        sorted_indices = np.argsort(scores)
        n_exploit = max(1, int(self.exploit_fraction * len(scores)))

        worst_indices = sorted_indices[:n_exploit]
        best_indices = sorted_indices[-n_exploit:]

        for worst_idx in worst_indices:
            if self.truncation_selection:
                best_idx = np.random.choice(best_indices)

                self.population[worst_idx].hyperparams = self.population[
                    best_idx
                ].hyperparams.copy()

                if self.population[best_idx].model_state is not None:
                    workers[worst_idx].load_state(self.population[best_idx].model_state)
                    self.population[worst_idx].model_state = self.population[
                        best_idx
                    ].model_state

                self.population[worst_idx].performance_history = []
                self.population[worst_idx].metadata["generation_created"] = (
                    self.generation
                )

            self._perturb_hyperparams(worst_idx)

    def _perturb_hyperparams(self, worker_idx: int) -> None:
        """
        Perturb hyperparameters for exploration.

        Parameters
        ----------
        worker_idx : int
            Index of worker to perturb
        """
        hyperparams = self.population[worker_idx].hyperparams

        param_names = list(hyperparams.keys())
        n_perturb = max(1, int(self.explore_fraction * len(param_names)))
        params_to_perturb = np.random.choice(param_names, n_perturb, replace=False)

        for param in params_to_perturb:
            if param in self.hyperparam_distributions:
                dist = self.hyperparam_distributions[param]
                hyperparams[param] = dist.perturb(hyperparams[param])

    def train(
        self,
        max_steps: int = 10000,
        max_generations: int = 100,
        timeout: Optional[float] = None,
        verbose: bool = True,
    ) -> PBTResult:
        """
        Run Population-Based Training.

        Parameters
        ----------
        max_steps : int, default=10000
            Maximum total training steps
        max_generations : int, default=100
            Maximum number of generations
        timeout : float, optional
            Maximum training time in seconds
        verbose : bool, default=True
            Whether to print progress information

        Returns
        -------
        PBTResult
            Training results including best worker and population history
        """
        if verbose:
            print("Starting Population-Based Training...")
            print(f"Population size: {self.population_size}")
            print(f"Evaluation interval: {self.eval_interval}")
            print(f"Exploit fraction: {self.exploit_fraction}")
            print(f"Explore fraction: {self.explore_fraction}")

        start_time = time.time()

        self._initialize_population()

        workers = [self.worker_factory() for _ in range(self.population_size)]

        for i, worker in enumerate(workers):
            worker.reset()

        best_score = -np.inf
        best_worker = None

        while (
            self.total_steps < max_steps
            and self.generation < max_generations
            and (timeout is None or time.time() - start_time < timeout)
        ):
            scores = self._evaluate_population(workers)
            self.total_steps += self.population_size * self.eval_interval

            valid_scores = [s for s in scores if s != -np.inf]
            if valid_scores:
                max_score_idx = np.argmax(scores)
                if scores[max_score_idx] > best_score:
                    best_score = scores[max_score_idx]
                    best_worker = copy.deepcopy(self.population[max_score_idx])

            population_snapshot = copy.deepcopy(self.population)
            self.population_history.append(population_snapshot)

            if verbose:
                if valid_scores:
                    mean_score = np.mean(valid_scores)
                    std_score = np.std(valid_scores) if len(valid_scores) > 1 else 0.0
                else:
                    mean_score = float("nan")
                    std_score = float("nan")

                print(
                    f"Generation {self.generation}: "
                    f"Best={best_score:.6f}, "
                    f"Mean={mean_score:.6f}±{std_score:.6f}, "
                    f"Steps={self.total_steps}"
                )

            self._exploit_and_explore(workers, scores)

            self.generation += 1

        total_time = time.time() - start_time

        if best_worker is None and self.population:
            best_overall_score = -np.inf
            for worker in self.population:
                if worker.performance_history:
                    worker_best = max(worker.performance_history)
                    if worker_best > best_overall_score:
                        best_overall_score = worker_best
                        best_worker = copy.deepcopy(worker)

        if best_worker is None and self.population:
            best_worker = copy.deepcopy(self.population[0])
            best_score = -np.inf

        final_scores = [
            (
                np.max(worker.performance_history)
                if worker.performance_history
                else -np.inf
            )
            for worker in self.population
        ]

        valid_final_scores = [s for s in final_scores if s != -np.inf]

        statistics = {
            "generations_completed": self.generation,
            "total_training_time": total_time,
            "final_population_mean": (
                np.mean(valid_final_scores) if valid_final_scores else float("nan")
            ),
            "final_population_std": (
                np.std(valid_final_scores) if len(valid_final_scores) > 1 else 0.0
            ),
            "best_score_progression": [
                max(
                    [
                        max(w.performance_history) if w.performance_history else -np.inf
                        for w in gen
                    ]
                )
                for gen in self.population_history
            ],
            "population_diversity": self._compute_diversity_metrics(),
            "convergence_generation": self._find_convergence_generation(),
        }

        if verbose:
            print(f"\nTraining completed in {total_time:.2f} seconds!")
            print(f"Generations: {self.generation}")
            print(f"Total steps: {self.total_steps}")
            print(f"Best score: {best_score:.6f}")
            if best_worker is not None:
                print(f"Best hyperparameters: {best_worker.hyperparams}")
            else:
                print("No valid workers found during training")

        return PBTResult(
            best_worker=best_worker,  # type: ignore
            final_population=self.population,
            population_history=self.population_history,
            total_training_time=total_time,
            total_steps=self.total_steps,
            statistics=statistics,
        )

    def _compute_diversity_metrics(self) -> Dict[str, float]:
        """Compute population diversity metrics."""
        if not self.population_history:
            return {}

        diversity_over_time = []

        for generation in self.population_history:
            param_diversities = []

            for param_name in self.hyperparam_distributions.keys():
                values = []
                for worker in generation:
                    if param_name in worker.hyperparams:
                        val = worker.hyperparams[param_name]
                        if isinstance(val, (int, float)):
                            values.append(val)

                if len(values) > 1:
                    diversity = np.std(values) / (np.mean(values) + 1e-8)
                    param_diversities.append(diversity)

            if param_diversities:
                diversity_over_time.append(np.mean(param_diversities))

        return {
            "initial_diversity": diversity_over_time[0] if diversity_over_time else 0.0,
            "final_diversity": diversity_over_time[-1] if diversity_over_time else 0.0,
            "mean_diversity": (  # type: ignore
                np.mean(diversity_over_time) if diversity_over_time else 0.0
            ),
            "diversity_trend": (
                diversity_over_time[-1] - diversity_over_time[0]
                if len(diversity_over_time) > 1
                else 0.0
            ),
        }

    def _find_convergence_generation(self) -> Optional[int]:
        """Find the generation where population converged."""
        if len(self.population_history) < 5:
            return None

        best_scores = []
        for generation in self.population_history:
            scores = [
                max(w.performance_history) if w.performance_history else -np.inf
                for w in generation
            ]
            best_scores.append(max(scores))

        improvement_threshold = 0.001
        window_size = 5

        for i in range(window_size, len(best_scores)):
            recent_improvement = (
                best_scores[i] - best_scores[i - window_size]
            ) / window_size
            if recent_improvement < improvement_threshold:
                return i

        return None


def pbt_optimize(
    train_function: Callable[[Dict, int], Tuple[float, Any]],
    save_function: Callable[[], Any],
    load_function: Callable[[Any], None],
    reset_function: Callable[[], None],
    initial_hyperparams: List[Dict[str, Any]],
    hyperparam_distributions: Dict[str, HyperparameterDistribution],
    population_size: int = 10,
    max_steps: int = 10000,
    eval_interval: int = 100,
    exploit_fraction: float = 0.2,
    explore_fraction: float = 0.2,
    random_state: Optional[int] = None,
    verbose: bool = True,
) -> PBTResult:
    """
    Convenience function for Population-Based Training.

    Parameters
    ----------
    train_function : callable
        Function that takes (hyperparams, steps) and returns (score, state)
    save_function : callable
        Function that returns current model state
    load_function : callable
        Function that loads given model state
    reset_function : callable
        Function that resets model to initial state
    initial_hyperparams : list
        Initial hyperparameter configurations
    hyperparam_distributions : dict
        Hyperparameter perturbation distributions
    population_size : int, default=10
        Size of population
    max_steps : int, default=10000
        Maximum training steps
    eval_interval : int, default=100
        Steps between evaluations
    exploit_fraction : float, default=0.2
        Fraction to exploit
    explore_fraction : float, default=0.2
        Fraction to explore
    random_state : int, optional
        Random seed
    verbose : bool, default=True
        Whether to print progress

    Returns
    -------
    PBTResult
        Training results

    Examples
    --------
    >>> # Define training functions
    >>> def train_step(hyperparams, steps):
    ...     # Simulate training
    ...     lr = hyperparams["learning_rate"]
    ...     # Performance improves with more steps but depends on lr
    ...     performance = 0.8 - (lr - 0.001) ** 2 + steps * 0.001
    ...     return performance, {"step": steps}
    >>>
    >>> def save_state():
    ...     return getattr(save_state, "state", {})
    >>>
    >>> def load_state(state):
    ...     save_state.state = state
    >>>
    >>> def reset():
    ...     save_state.state = {}
    >>>
    >>> # Define hyperparameters
    >>> initial_configs = [
    ...     {"learning_rate": 0.001},
    ...     {"learning_rate": 0.01},
    ...     {"learning_rate": 0.0001},
    ... ]
    >>>
    >>> distributions = {
    ...     "learning_rate": LogUniformPerturbation((0.8, 1.2), (1e-5, 1e-1))
    ... }
    >>>
    >>> result = pbt_optimize(
    ...     train_step,
    ...     save_state,
    ...     load_state,
    ...     reset,
    ...     initial_configs,
    ...     distributions,
    ...     population_size=5,
    ... )
    """

    def worker_factory():
        return FunctionWorker(
            train_function, save_function, load_function, reset_function
        )

    trainer = PopulationBasedTrainer(
        worker_factory=worker_factory,
        initial_hyperparams=initial_hyperparams,
        hyperparam_distributions=hyperparam_distributions,
        population_size=population_size,
        eval_interval=eval_interval,
        exploit_fraction=exploit_fraction,
        explore_fraction=explore_fraction,
        random_state=random_state,
    )

    return trainer.train(max_steps=max_steps, verbose=verbose)


if __name__ == "__main__":
    print("Example 1: Simple Optimization Problem")

    worker_states_simple = {}

    def simple_train_step(hyperparams: Dict[str, Any], steps: int) -> Tuple[float, Any]:
        """Simulate training step with time-varying optimal hyperparameters."""
        worker_id = threading.current_thread().ident

        if worker_id not in worker_states_simple:
            worker_states_simple[worker_id] = {"total_steps": 0, "momentum": 0.0}

        state = worker_states_simple[worker_id]
        lr = hyperparams["learning_rate"]

        total_steps = state["total_steps"]
        optimal_lr = 0.01 * np.exp(-total_steps / 1000)  # Decreasing over time

        base_performance = 0.9 - (lr - optimal_lr) ** 2 * 100

        momentum_effect = state["momentum"] * 0.1
        performance = base_performance + momentum_effect

        state["total_steps"] += steps
        state["momentum"] = 0.9 * state["momentum"] + 0.1 * performance

        performance += np.random.normal(0, 0.01)
        return performance, state.copy()

    def simple_save_state():
        """Save current worker state."""
        worker_id = threading.current_thread().ident
        return worker_states_simple.get(worker_id, {}).copy()

    def simple_load_state(state):
        """Load worker state."""
        worker_id = threading.current_thread().ident
        worker_states_simple[worker_id] = state.copy()

    def simple_reset():
        """Reset worker state."""
        worker_id = threading.current_thread().ident
        worker_states_simple[worker_id] = {"total_steps": 0, "momentum": 0.0}

    initial_configs = [
        {"learning_rate": 0.001},
        {"learning_rate": 0.01},
        {"learning_rate": 0.1},
        {"learning_rate": 0.0001},
    ]

    distributions = {
        "learning_rate": LogUniformPerturbation(
            factor_range=(0.8, 1.25), bounds=(1e-5, 1.0)
        )
    }

    result1 = pbt_optimize(
        train_function=simple_train_step,
        save_function=simple_save_state,
        load_function=simple_load_state,
        reset_function=simple_reset,
        initial_hyperparams=initial_configs,
        hyperparam_distributions=distributions,  # type: ignore
        population_size=6,
        max_steps=3000,
        eval_interval=50,
        exploit_fraction=0.25,
        explore_fraction=0.3,
        random_state=42,
        verbose=True,
    )

    print(f"\nBest configuration found:")
    print(f"  Learning rate: {result1.best_worker.hyperparams['learning_rate']:.6f}")
    print(f"  Best score: {max(result1.best_worker.performance_history):.6f}")
    print(f"  Training steps: {result1.best_worker.training_step}")

    print("\n" + "=" * 60)
    print("Example 2: Multi-Hyperparameter Neural Network")

    worker_states_nn = {}

    def nn_train_step(hyperparams: Dict[str, Any], steps: int) -> Tuple[float, Any]:
        """Simulate neural network training with multiple hyperparameters."""
        worker_id = threading.current_thread().ident

        if worker_id not in worker_states_nn:
            worker_states_nn[worker_id] = {
                "total_steps": 0,
                "validation_history": [],
                "overfitting_penalty": 0.0,
            }

        state = worker_states_nn[worker_id]

        lr = hyperparams["learning_rate"]
        wd = hyperparams["weight_decay"]
        batch_size = hyperparams["batch_size"]
        dropout = hyperparams["dropout_rate"]

        total_steps = state["total_steps"]
        base_performance = 0.85 * (1 - np.exp(-total_steps / 500))

        lr_effect = -2 * (np.log10(lr) + 3) ** 2  # Optimal around 1e-3
        wd_effect = -0.5 * (np.log10(wd) + 4) ** 2  # Optimal around 1e-4
        batch_effect = -0.001 * (batch_size - 64) ** 2  # Optimal around 64
        dropout_effect = -2 * (dropout - 0.2) ** 2  # Optimal around 0.2

        if total_steps > 800:
            overfitting = 0.1 * (total_steps - 800) / 1000
            overfitting_protection = dropout * 0.2
            state["overfitting_penalty"] = overfitting - overfitting_protection

        performance = (
            base_performance
            + lr_effect
            + wd_effect
            + batch_effect
            + dropout_effect
            - state.get("overfitting_penalty", 0.0)
        )

        state["total_steps"] += steps
        state["validation_history"].append(performance)

        if len(state["validation_history"]) > 20:
            state["validation_history"] = state["validation_history"][-20:]

        performance += np.random.normal(0, 0.01)
        return performance, state.copy()

    def nn_save_state():
        """Save current worker state."""
        worker_id = threading.current_thread().ident
        return worker_states_nn.get(
            worker_id,
            {
                "total_steps": 0,
                "validation_history": [],
                "overfitting_penalty": 0.0,
            },
        ).copy()

    def nn_load_state(state):
        """Load worker state."""
        worker_id = threading.current_thread().ident
        worker_states_nn[worker_id] = state.copy()

    def nn_reset():
        """Reset worker state."""
        worker_id = threading.current_thread().ident
        worker_states_nn[worker_id] = {
            "total_steps": 0,
            "validation_history": [],
            "overfitting_penalty": 0.0,
        }

    nn_configs = [
        {
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 64,
            "dropout_rate": 0.1,
        },
        {
            "learning_rate": 0.01,
            "weight_decay": 0.001,
            "batch_size": 32,
            "dropout_rate": 0.2,
        },
        {
            "learning_rate": 0.0001,
            "weight_decay": 0.00001,
            "batch_size": 128,
            "dropout_rate": 0.3,
        },
        {
            "learning_rate": 0.003,
            "weight_decay": 0.0003,
            "batch_size": 64,
            "dropout_rate": 0.15,
        },
        {
            "learning_rate": 0.005,
            "weight_decay": 0.0005,
            "batch_size": 96,
            "dropout_rate": 0.25,
        },
    ]

    nn_distributions = {
        "learning_rate": LogUniformPerturbation((0.8, 1.2), (1e-5, 1e-1)),
        "weight_decay": LogUniformPerturbation((0.8, 1.2), (1e-6, 1e-2)),
        "batch_size": ChoicePerturbation([16, 32, 64, 96, 128], 0.2),
        "dropout_rate": UniformPerturbation(0.05, (0.0, 0.5)),
    }

    result2 = pbt_optimize(
        train_function=nn_train_step,
        save_function=nn_save_state,
        load_function=nn_load_state,
        reset_function=nn_reset,
        initial_hyperparams=nn_configs,
        hyperparam_distributions=nn_distributions,
        population_size=8,
        max_steps=6000,
        eval_interval=100,
        exploit_fraction=0.2,
        explore_fraction=0.25,
        random_state=42,
        verbose=True,
    )

    print(f"\nBest neural network configuration:")
    if result2.best_worker is not None:
        for param, value in result2.best_worker.hyperparams.items():
            if isinstance(value, float):
                print(f"  {param}: {value:.6f}")
            else:
                print(f"  {param}: {value}")

        if result2.best_worker.performance_history:
            print(
                f"\nBest validation accuracy: {max(result2.best_worker.performance_history):.6f}"
            )
            print(f"Training steps: {result2.best_worker.training_step}")
        else:
            print("\nNo performance history available")
    else:
        print("No valid configuration found")

    print(f"\nPopulation diversity analysis:")
    for metric, value in result2.statistics["population_diversity"].items():
        if not np.isnan(value):
            print(f"  {metric}: {value:.4f}")

    if result2.statistics["convergence_generation"] is not None:
        print(
            f"\nPopulation converged at generation {result2.statistics['convergence_generation']}"
        )

    best_scores = result2.statistics["best_score_progression"]
    print(f"\nPerformance progression (every 5 generations):")
    for i in range(0, len(best_scores), 5):
        if not np.isneginf(best_scores[i]):
            print(f"  Generation {i}: {best_scores[i]:.6f}")

    print("\n" + "=" * 60)
    print("Example 3: Hyperparameter Trajectory Analysis")

    print(f"\nAnalyzing hyperparameter evolution...")

    if result2.best_worker is not None and result2.population_history:
        learning_rates_over_time = []
        for generation in result2.population_history:
            valid_workers = [w for w in generation if w.performance_history]
            if valid_workers:
                best_worker_gen = max(
                    valid_workers,
                    key=lambda w: max(w.performance_history),
                )
                learning_rates_over_time.append(
                    best_worker_gen.hyperparams["learning_rate"]
                )

        if learning_rates_over_time:
            print(f"\nBest worker's learning rate trajectory:")
            print(f"  Initial: {learning_rates_over_time[0]:.6f}")
            print(f"  Final: {learning_rates_over_time[-1]:.6f}")
            print(f"  Min: {min(learning_rates_over_time):.6f}")
            print(f"  Max: {max(learning_rates_over_time):.6f}")
            print(f"  Mean: {np.mean(learning_rates_over_time):.6f}")

            print(f"\nEvery 10th generation:")
            for i in range(0, len(learning_rates_over_time), 10):
                print(f"  Generation {i}: LR = {learning_rates_over_time[i]:.6f}")
        else:
            print("No valid learning rate trajectory found")
    else:
        print("No valid trajectory data available")

    print("\n" + "=" * 60)
    print("Example 4: PBT vs Fixed Hyperparameters")

    worker_states_fixed = {}

    def fixed_nn_train_step(
        hyperparams: Dict[str, Any], steps: int
    ) -> Tuple[float, Any]:
        """Simulate neural network training with fixed hyperparameters."""
        worker_id = threading.current_thread().ident

        if worker_id not in worker_states_fixed:
            worker_states_fixed[worker_id] = {
                "total_steps": 0,
                "validation_history": [],
                "overfitting_penalty": 0.0,
            }

        state = worker_states_fixed[worker_id]

        lr = hyperparams["learning_rate"]
        wd = hyperparams["weight_decay"]
        batch_size = hyperparams["batch_size"]
        dropout = hyperparams["dropout_rate"]

        total_steps = state["total_steps"]
        base_performance = 0.85 * (1 - np.exp(-total_steps / 500))

        lr_effect = -2 * (np.log10(lr) + 3) ** 2
        wd_effect = -0.5 * (np.log10(wd) + 4) ** 2
        batch_effect = -0.001 * (batch_size - 64) ** 2
        dropout_effect = -2 * (dropout - 0.2) ** 2

        if total_steps > 800:
            overfitting = 0.1 * (total_steps - 800) / 1000
            overfitting_protection = dropout * 0.2
            state["overfitting_penalty"] = overfitting - overfitting_protection

        performance = (
            base_performance
            + lr_effect
            + wd_effect
            + batch_effect
            + dropout_effect
            - state.get("overfitting_penalty", 0.0)
        )

        state["total_steps"] += steps
        state["validation_history"].append(performance)

        if len(state["validation_history"]) > 20:
            state["validation_history"] = state["validation_history"][-20:]

        performance += np.random.normal(0, 0.01)
        return performance, state.copy()

    fixed_config = {
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 64,
        "dropout_rate": 0.2,
    }

    print(f"\nTraining with fixed hyperparameters: {fixed_config}")

    total_fixed_steps = (
        result2.total_steps // result2.statistics["generations_completed"]
    )
    fixed_scores = []

    for step in range(0, total_fixed_steps, 100):
        score, _ = fixed_nn_train_step(fixed_config, 100)
        fixed_scores.append(score)

    fixed_best = max(fixed_scores) if fixed_scores else -np.inf

    if result2.best_worker is not None and result2.best_worker.performance_history:
        pbt_best = max(result2.best_worker.performance_history)
    else:
        pbt_best = -np.inf

    print(f"\nResults comparison:")
    print(f"  PBT best score: {pbt_best:.6f}")
    print(f"  Fixed hyperparams best score: {fixed_best:.6f}")
    if pbt_best > -np.inf and fixed_best > -np.inf:
        print(
            f"  Improvement: {(pbt_best - fixed_best):.6f} ({100 * (pbt_best - fixed_best) / fixed_best:.2f}%)"
        )

    if pbt_best > fixed_best:
        print(f"\nPBT found better hyperparameters through online adaptation!")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nPopulation-Based Training successfully:")
    print(f"  - Trained {result2.statistics['generations_completed']} generations")
    print(f"  - Evaluated {result2.total_steps} total training steps")
    print(f"  - Adapted hyperparameters online during training")
    if result2.best_worker is not None:
        print(f"  - Found optimal configuration: {result2.best_worker.hyperparams}")
        if pbt_best > -np.inf:
            print(f"  - Achieved best score: {pbt_best:.6f}")

    print(f"\nKey insights:")
    diversity_stats = result2.statistics["population_diversity"]
    if not np.isnan(diversity_stats.get("initial_diversity", float("nan"))):
        print(
            f"  - Population diversity changed from {diversity_stats['initial_diversity']:.4f} to {diversity_stats['final_diversity']:.4f}"
        )
    if (
        best_scores
        and not np.isneginf(best_scores[0])
        and not np.isneginf(best_scores[-1])
    ):
        print(
            f"  - Best score improved from {best_scores[0]:.6f} to {best_scores[-1]:.6f}"
        )
    print(
        f"  - Hyperparameters adapted {result2.statistics['generations_completed']} times during training"
    )

    print("\nPopulation-Based Training examples completed!")
