"""
Multi-Fidelity Optimization for Hyperparameter Tuning
=====================================================

ASHA (Asynchronous Successive Halving) implementation for efficient resource
allocation across hyperparameter candidates. This approach uses cheaper
approximations (lower fidelity) to guide the search, then evaluates promising
candidates at full fidelity.

References
----------
- Li, L., et al. (2018). "Massively Parallel Hyperparameter Tuning."
  arXiv preprint arXiv:1810.05934.
- Jamieson, K., & Talwalkar, A. (2016). "Non-stochastic best arm identification
  and hyperparameter optimization." Artificial Intelligence and Statistics.

Author
------
Deep Learning Reference Hub

License
-------
MIT License

Notes
-----
Multi-fidelity optimization is particularly effective when:
1. Training time is expensive
2. Early performance correlates with final performance
3. You have many hyperparameter configurations to evaluate
4. Computational resources can be allocated dynamically
"""

import threading
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FidelityConfig:
    """
    Configuration for a fidelity level.

    Attributes
    ----------
    name : str
        Name of the fidelity level
    budget : int
        Budget/resource allocation for this fidelity
    min_budget : int
        Minimum budget required for this fidelity
    max_budget : int
        Maximum budget for this fidelity
    """

    name: str
    budget: int
    min_budget: int = 1
    max_budget: int = 1000


@dataclass
class CandidateResult:
    """
    Result from evaluating a hyperparameter candidate.

    Attributes
    ----------
    config_id : int
        Unique identifier for the configuration
    hyperparams : dict
        Hyperparameter configuration
    fidelity : int
        Fidelity level used for evaluation
    score : float
        Performance score achieved
    training_time : float
        Time taken for training
    metadata : dict
        Additional metadata from training
    """

    config_id: int
    hyperparams: dict[str, Any]
    fidelity: int
    score: float
    training_time: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiFidelityResult:
    """
    Results from multi-fidelity optimization.

    Attributes
    ----------
    best_config : dict
        Best hyperparameter configuration found
    best_score : float
        Best score achieved
    best_fidelity : int
        Fidelity level of best result
    all_results : list
        All evaluation results
    total_time : float
        Total optimization time
    total_budget_used : int
        Total computational budget consumed
    statistics : dict
        Optimization statistics and analysis
    """

    best_config: dict[str, Any]
    best_score: float
    best_fidelity: int
    all_results: list[CandidateResult]
    total_time: float
    total_budget_used: int
    statistics: dict[str, Any] = field(default_factory=dict)


class FidelityEvaluator(ABC):
    """
    Abstract base class for fidelity-aware evaluation.

    Defines the interface for evaluating hyperparameter configurations
    at different fidelity levels.
    """

    @abstractmethod
    def evaluate(
        self, hyperparams: dict[str, Any], fidelity: int
    ) -> tuple[float, dict[str, Any]]:
        """
        Evaluate hyperparameters at given fidelity.

        Parameters
        ----------
        hyperparams : dict
            Hyperparameter configuration
        fidelity : int
            Fidelity level (e.g., training epochs, data size)

        Returns
        -------
        tuple
            (score, metadata) where score is performance and metadata contains
            additional information from training
        """
        pass

    @abstractmethod
    def get_fidelity_range(self) -> tuple[int, int]:
        """
        Get the valid fidelity range.

        Returns
        -------
        tuple
            (min_fidelity, max_fidelity)
        """
        pass


class FunctionEvaluator(FidelityEvaluator):
    """
    Function-based evaluator wrapper.

    Wraps a user-provided evaluation function to conform to the
    FidelityEvaluator interface.

    Parameters
    ----------
    eval_function : callable
        Function that takes (hyperparams, fidelity) and returns (score, metadata)
    min_fidelity : int
        Minimum fidelity level
    max_fidelity : int
        Maximum fidelity level
    """

    def __init__(
        self, eval_function: Callable, min_fidelity: int = 1, max_fidelity: int = 100
    ):
        self.eval_function = eval_function
        self.min_fidelity = min_fidelity
        self.max_fidelity = max_fidelity

    def evaluate(
        self, hyperparams: dict[str, Any], fidelity: int
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate using wrapped function."""
        return self.eval_function(hyperparams, fidelity)

    def get_fidelity_range(self) -> tuple[int, int]:
        """Get fidelity range."""
        return self.min_fidelity, self.max_fidelity


class ASHAOptimizer:
    """
    Asynchronous Successive Halving Algorithm (ASHA) for multi-fidelity optimization.

    ASHA efficiently allocates computational resources by starting many configurations
    at low fidelity and promoting the most promising ones to higher fidelities.

    Parameters
    ----------
    evaluator : FidelityEvaluator
        Evaluator for hyperparameter configurations
    reduction_factor : int, default=3
        Factor by which to reduce number of configurations at each rung
    min_budget : int, default=1
        Minimum budget (fidelity) to start configurations
    max_budget : int, default=81
        Maximum budget (fidelity) for full evaluation
    grace_period : int, default=1
        Minimum budget before first promotion opportunity
    max_concurrent : int, default=4
        Maximum number of concurrent evaluations
    random_state : int, optional
        Random seed for reproducibility
    """

    def __init__(
        self,
        evaluator: FidelityEvaluator,
        reduction_factor: int = 3,
        min_budget: int = 1,
        max_budget: int = 81,
        grace_period: int = 1,
        max_concurrent: int = 4,
        random_state: int | None = None,
    ):

        self.evaluator = evaluator
        self.reduction_factor = reduction_factor
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.grace_period = grace_period
        self.max_concurrent = max_concurrent

        if random_state is not None:
            np.random.seed(random_state)

        # Validate parameters
        eval_min, eval_max = evaluator.get_fidelity_range()
        if min_budget < eval_min or max_budget > eval_max:
            raise ValueError(
                f"Budget range [{min_budget}, {max_budget}] "
                f"outside evaluator range [{eval_min}, {eval_max}]"
            )

        # Initialize internal state
        self.config_counter = 0
        self.rungs = self._create_rungs()
        self.results_history = []
        self.active_evaluations = {}
        self.lock = threading.Lock()

        # Statistics tracking
        self.total_budget_used = 0
        self.best_result = None

    def _create_rungs(self) -> list[dict]:
        """
        Create ASHA rungs (fidelity levels and promotion thresholds).

        Returns
        -------
        list
            List of rung dictionaries with budget and promotion info
        """
        rungs = []
        current_budget = self.min_budget

        while current_budget <= self.max_budget:
            rung = {
                "budget": current_budget,
                "candidates": [],  # (config_id, score) tuples
                "promoted": set(),  # Set of promoted config_ids
                "n_required": 0,  # Number of configs needed for promotion
            }
            rungs.append(rung)
            current_budget *= self.reduction_factor

        # A rung promotes its top 1 / reduction_factor, so it needs at least
        # reduction_factor results before that fraction means anything. The
        # threshold is a property of the reduction factor, not of how many rungs
        # the ladder happens to have: deriving it from the rung count made a
        # taller ladder demand more results at the bottom and stall there.
        for rung in rungs[:-1]:  # The last rung promotes nowhere
            rung["n_required"] = self.reduction_factor

        return rungs

    def _get_rung_for_budget(self, budget: int) -> int | None:
        """Get rung index for given budget."""
        for i, rung in enumerate(self.rungs):
            if rung["budget"] == budget:
                return i
        return None

    def _add_result(self, result: CandidateResult) -> None:
        """Add result and check for promotions."""
        with self.lock:
            self.results_history.append(result)
            self.total_budget_used += result.fidelity

            if self.best_result is None or result.score > self.best_result.score:
                self.best_result = result

            rung_idx = self._get_rung_for_budget(result.fidelity)
            if rung_idx is not None:
                rung = self.rungs[rung_idx]
                rung["candidates"].append((result.config_id, result.score))

    def _get_next_config_to_evaluate(self) -> tuple[int, dict[str, Any], int] | None:
        """
        Claim the next promotion and return the work it implies.

        This is the only place the promotion rule lives, and calling it is not a
        query: the returned configuration is recorded as promoted so that
        concurrent workers cannot claim it twice. Callers must therefore use
        what they get rather than calling it to ask whether work exists.

        Returns
        -------
        tuple or None
            (config_id, hyperparams, fidelity) or None if no work available
        """
        with self.lock:
            for rung_idx in range(len(self.rungs) - 1):
                rung = self.rungs[rung_idx]
                next_rung = self.rungs[rung_idx + 1]

                if len(rung["candidates"]) >= rung["n_required"]:
                    sorted_candidates = sorted(
                        rung["candidates"], key=lambda x: x[1], reverse=True
                    )
                    n_promote = max(1, len(sorted_candidates) // self.reduction_factor)

                    for i in range(min(n_promote, len(sorted_candidates))):
                        config_id, score = sorted_candidates[i]

                        if config_id not in rung["promoted"]:
                            rung["promoted"].add(config_id)

                            hyperparams = None
                            for res in self.results_history:
                                if res.config_id == config_id:
                                    hyperparams = res.hyperparams
                                    break

                            if hyperparams is not None:
                                return config_id, hyperparams, next_rung["budget"]

            return None

    def _evaluate_config(
        self, config_id: int, hyperparams: dict[str, Any], fidelity: int
    ) -> CandidateResult:
        """Evaluate a single configuration."""
        start_time = time.time()

        try:
            score, metadata = self.evaluator.evaluate(hyperparams, fidelity)
            training_time = time.time() - start_time

            if np.isnan(score) or np.isinf(score):
                score = -np.inf

            return CandidateResult(
                config_id=config_id,
                hyperparams=hyperparams.copy(),
                fidelity=fidelity,
                score=float(score),
                training_time=training_time,
                metadata=metadata,
            )

        except Exception as e:
            training_time = time.time() - start_time
            warnings.warn(f"Evaluation failed for config {config_id}: {e}")

            return CandidateResult(
                config_id=config_id,
                hyperparams=hyperparams.copy(),
                fidelity=fidelity,
                score=-np.inf,
                training_time=training_time,
                metadata={"error": str(e)},
            )

    def suggest_initial_configurations(
        self, configurations: list[dict[str, Any]]
    ) -> None:
        """
        Add initial configurations to start evaluation.

        Parameters
        ----------
        configurations : list
            List of hyperparameter configurations to evaluate
        """
        with self.lock:
            for config in configurations:
                config_id = self.config_counter
                self.config_counter += 1
                # Queued at the lowest rung. Every configuration earns its way
                # up from here, which is what makes the search cheap.
                self.active_evaluations[config_id] = (config.copy(), self.min_budget)

    def optimize(
        self,
        initial_configurations: list[dict[str, Any]],
        max_iterations: int = 100,
        timeout: float | None = None,
        verbose: bool = True,
    ) -> MultiFidelityResult:
        """
        Run ASHA optimization.

        Parameters
        ----------
        initial_configurations : list
            Initial hyperparameter configurations to evaluate
        max_iterations : int, default=100
            Maximum number of evaluations to perform. Counted at submission, and
            every submitted evaluation is recorded, so this bounds the results as
            well as the work -- at any `max_concurrent`.
        timeout : float, optional
            Maximum time in seconds (None for no timeout). Evaluations already
            running when the clock runs out are still awaited and recorded; the
            timeout stops new submissions, it does not cancel paid-for work.
        verbose : bool, default=True
            Whether to print progress information

        Returns
        -------
        MultiFidelityResult
            Optimization results
        """
        if verbose:
            print("Starting ASHA Multi-Fidelity Optimization...")
            print(f"Reduction factor: {self.reduction_factor}")
            print(f"Budget range: [{self.min_budget}, {self.max_budget}]")
            print(f"Initial configurations: {len(initial_configurations)}")
            print(f"Max concurrent evaluations: {self.max_concurrent}")

        start_time = time.time()
        self.suggest_initial_configurations(initial_configurations)

        iteration = 0
        evaluations_completed = 0

        work_queue = []
        for config_id, (config, fidelity) in self.active_evaluations.items():
            work_queue.append((config_id, config, fidelity))
        self.active_evaluations.clear()

        def record(future, config_id: int) -> None:
            """Move one finished evaluation into the results, or warn."""
            nonlocal evaluations_completed

            try:
                result = future.result()
            except Exception as e:
                warnings.warn(f"Future failed for config {config_id}: {e}")
                return

            self._add_result(result)
            evaluations_completed += 1

            if verbose and evaluations_completed % max(1, max_iterations // 20) == 0:
                print(
                    f"Completed {evaluations_completed} evaluations - "
                    f"Best score: {self.best_result.score:.6f} "  # type: ignore
                    f"(fidelity {self.best_result.fidelity})"  # type: ignore
                )

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            active_futures = {}

            while iteration < max_iterations and (
                timeout is None or time.time() - start_time < timeout
            ):
                # `_get_next_config_to_evaluate` marks what it returns as
                # promoted, so it is called once per work item and its result is
                # kept. Calling it again as a loop condition would consume a
                # promotion and drop the configuration on the floor.
                while len(active_futures) < self.max_concurrent:
                    if work_queue:
                        config_id, hyperparams, fidelity = work_queue.pop(0)
                    else:
                        next_work = self._get_next_config_to_evaluate()
                        if next_work is None:
                            break
                        config_id, hyperparams, fidelity = next_work

                    # Submit evaluation
                    future = executor.submit(
                        self._evaluate_config, config_id, hyperparams, fidelity
                    )
                    active_futures[future] = (config_id, hyperparams, fidelity)
                    iteration += 1

                    if iteration >= max_iterations:
                        break

                if active_futures:
                    # The one-second bound is a poll interval, not a deadline: it
                    # returns control to the loop so `max_iterations` and
                    # `timeout` get re-checked while evaluations are still
                    # running. `as_completed` reports an elapsed poll by raising,
                    # and a tuner exists to run evaluations that take minutes, so
                    # letting that escape would abort every realistic run the
                    # moment no evaluation happened to finish within a second.
                    # Before 3.11 this is not the builtin `TimeoutError`, so it
                    # is caught under its own name rather than by coincidence.
                    completed_futures = []
                    try:
                        for future in as_completed(active_futures, timeout=1.0):
                            completed_futures.append(future)
                            break  # Process one at a time for responsiveness
                    except FuturesTimeoutError:
                        pass

                    for future in completed_futures:
                        config_id, _, _ = active_futures.pop(future)
                        record(future, config_id)

                # Break if no more work and no active evaluations
                if not active_futures and not work_queue:
                    # Requeued rather than discarded: this call promotes the
                    # configuration it returns, so dropping it would lose the
                    # promotion and end the run one rung short.
                    next_work = self._get_next_config_to_evaluate()
                    if next_work is None:
                        if verbose:
                            print("No more configurations to evaluate - stopping")
                        break
                    work_queue.append(next_work)

            # Whatever is still in flight when the loop exits has already been
            # submitted, so the pool will compute it whether or not anyone waits:
            # `ThreadPoolExecutor.__exit__` joins every worker. Recording it is
            # therefore free, and discarding it would under-report
            # `total_budget_used` by up to `max_concurrent - 1` evaluations --
            # flattering `budget_efficiency` with work the run really paid for.
            # This also keeps `max_iterations` meaning one thing at any worker
            # count: submissions, which now all become results.
            for future, (config_id, _, _) in list(active_futures.items()):
                record(future, config_id)
            active_futures.clear()

        total_time = time.time() - start_time

        statistics = self._compute_statistics()

        if verbose:
            print(f"\nOptimization completed in {total_time:.2f} seconds!")
            print(f"Total evaluations: {evaluations_completed}")
            print(f"Total budget used: {self.total_budget_used}")
            print(f"Best score: {self.best_result.score:.6f}")  # type: ignore
            print(f"Best configuration: {self.best_result.hyperparams}")  # type: ignore
            print(f"Best fidelity: {self.best_result.fidelity}")  # type: ignore

        return MultiFidelityResult(
            best_config=self.best_result.hyperparams,  # type: ignore
            best_score=self.best_result.score,  # type: ignore
            best_fidelity=self.best_result.fidelity,  # type: ignore
            all_results=self.results_history,
            total_time=total_time,
            total_budget_used=self.total_budget_used,
            statistics=statistics,
        )

    def _compute_statistics(self) -> dict[str, Any]:
        """Compute optimization statistics."""
        if not self.results_history:
            return {}

        fidelity_stats = defaultdict(list)
        for result in self.results_history:
            if result.score != -np.inf:
                fidelity_stats[result.fidelity].append(result.score)

        fidelity_analysis = {}
        for fidelity, scores in fidelity_stats.items():
            fidelity_analysis[fidelity] = {
                "n_evaluations": len(scores),
                "mean_score": np.mean(scores),
                "std_score": np.std(scores),
                "max_score": np.max(scores),
                "min_score": np.min(scores),
            }

        all_scores = [r.score for r in self.results_history if r.score != -np.inf]
        all_times = [r.training_time for r in self.results_history]

        statistics = {
            "total_evaluations": len(self.results_history),
            "successful_evaluations": len(all_scores),
            "failed_evaluations": len(self.results_history) - len(all_scores),
            "mean_score": np.mean(all_scores) if all_scores else 0.0,
            "std_score": np.std(all_scores) if len(all_scores) > 1 else 0.0,
            "mean_training_time": np.mean(all_times),
            "total_training_time": np.sum(all_times),
            "fidelity_analysis": fidelity_analysis,
            "budget_efficiency": (
                self.best_result.score / self.total_budget_used  # type: ignore
                if self.total_budget_used > 0
                else 0.0
            ),
            "rungs_used": len([r for r in self.rungs if r["candidates"]]),
        }

        return statistics


def asha_optimize(
    eval_function: Callable[[dict, int], tuple[float, dict]],
    initial_configurations: list[dict[str, Any]],
    min_fidelity: int = 1,
    max_fidelity: int = 81,
    reduction_factor: int = 3,
    max_iterations: int = 100,
    max_concurrent: int = 4,
    timeout: float | None = None,
    random_state: int | None = None,
    verbose: bool = True,
) -> MultiFidelityResult:
    """
    Convenience function for ASHA optimization.

    Parameters
    ----------
    eval_function : callable
        Function that takes (hyperparams, fidelity) and returns (score, metadata)
    initial_configurations : list
        Initial hyperparameter configurations to evaluate
    min_fidelity : int, default=1
        Minimum fidelity level
    max_fidelity : int, default=81
        Maximum fidelity level
    reduction_factor : int, default=3
        ASHA reduction factor
    max_iterations : int, default=100
        Maximum number of evaluations, performed and recorded alike
    max_concurrent : int, default=4
        Maximum concurrent evaluations
    timeout : float, optional
        Timeout in seconds; evaluations already running are still awaited
    random_state : int, optional
        Random seed
    verbose : bool, default=True
        Whether to print progress

    Returns
    -------
    MultiFidelityResult
        Optimization results

    Examples
    --------
    >>> def evaluate_model(hyperparams, fidelity):
    ...     # Simulate training with given hyperparameters and fidelity
    ...     lr = hyperparams["learning_rate"]
    ...     wd = hyperparams["weight_decay"]
    ...
    ...     # Simulate performance improving with fidelity
    ...     base_score = 0.7 - (lr - 0.001) ** 2 - (wd - 0.0001) ** 2
    ...     fidelity_bonus = 0.2 * (1 - np.exp(-fidelity / 20))
    ...     noise = np.random.normal(0, 0.01)
    ...
    ...     score = base_score + fidelity_bonus + noise
    ...     metadata = {"fidelity_used": fidelity}
    ...
    ...     return score, metadata
    >>>
    >>> configs = [
    ...     {"learning_rate": 0.001, "weight_decay": 0.0001},
    ...     {"learning_rate": 0.01, "weight_decay": 0.001},
    ...     {"learning_rate": 0.0001, "weight_decay": 0.00001},
    ... ]
    >>>
    >>> result = asha_optimize(
    ...     evaluate_model, configs, min_fidelity=1, max_fidelity=27, max_iterations=20
    ... )
    """
    evaluator = FunctionEvaluator(eval_function, min_fidelity, max_fidelity)

    optimizer = ASHAOptimizer(
        evaluator=evaluator,
        reduction_factor=reduction_factor,
        min_budget=min_fidelity,
        max_budget=max_fidelity,
        max_concurrent=max_concurrent,
        random_state=random_state,
    )

    return optimizer.optimize(
        initial_configurations=initial_configurations,
        max_iterations=max_iterations,
        timeout=timeout,
        verbose=verbose,
    )


def analyze_fidelity_correlation(result: MultiFidelityResult) -> dict[str, float]:
    """
    Analyze correlation between different fidelity levels.

    Parameters
    ----------
    result : MultiFidelityResult
        Results from multi-fidelity optimization

    Returns
    -------
    dict
        Correlation analysis between fidelity levels
    """
    config_results = defaultdict(dict)
    for res in result.all_results:
        if res.score != -np.inf:
            config_results[res.config_id][res.fidelity] = res.score

    fidelities = sorted(set(res.fidelity for res in result.all_results))
    correlations = {}

    for i, fid1 in enumerate(fidelities[:-1]):
        for fid2 in fidelities[i + 1 :]:
            common_configs = []
            scores1, scores2 = [], []

            for config_id, fid_scores in config_results.items():
                if fid1 in fid_scores and fid2 in fid_scores:
                    scores1.append(fid_scores[fid1])
                    scores2.append(fid_scores[fid2])

            if len(scores1) >= 3:  # Need at least 3 points for correlation
                correlation = np.corrcoef(scores1, scores2)[0, 1]
                if not np.isnan(correlation):
                    correlations[f"fidelity_{fid1}_vs_{fid2}"] = correlation

    return correlations


if __name__ == "__main__":
    print("Example 1: Quadratic Function with Fidelity")

    def quadratic_eval(
        hyperparams: dict[str, Any], fidelity: int
    ) -> tuple[float, dict[str, Any]]:
        """Simulate model evaluation with fidelity-dependent performance."""
        x, y = hyperparams["x"], hyperparams["y"]

        base_score = -((x - 2) ** 2 + (y + 1) ** 2)

        fidelity_bonus = 2 * (1 - np.exp(-fidelity / 10))

        # Add noise (decreases with fidelity)
        noise_std = 0.1 / np.sqrt(fidelity)
        noise = np.random.normal(0, noise_std)

        final_score = base_score + fidelity_bonus + noise
        metadata = {
            "base_score": base_score,
            "fidelity_bonus": fidelity_bonus,
            "noise": noise,
        }

        return final_score, metadata

    initial_configs = []
    for _ in range(15):
        config = {"x": np.random.uniform(-5, 5), "y": np.random.uniform(-5, 5)}
        initial_configs.append(config)

    print(f"Generated {len(initial_configs)} initial configurations")

    result1 = asha_optimize(
        eval_function=quadratic_eval,
        initial_configurations=initial_configs,
        min_fidelity=1,
        max_fidelity=27,  # 3^3
        reduction_factor=3,
        max_iterations=50,
        max_concurrent=3,
        random_state=42,
        verbose=True,
    )

    print(f"\nBest configuration found:")
    print(f"  x = {result1.best_config['x']:.4f}")
    print(f"  y = {result1.best_config['y']:.4f}")
    print(f"  Score = {result1.best_score:.4f}")
    print(f"  Fidelity = {result1.best_fidelity}")
    print(f"True optimum: x=2, y=-1")

    print("\n" + "=" * 60)
    print("Example 2: Neural Network Hyperparameter Optimization")

    def nn_eval(
        hyperparams: dict[str, Any], fidelity: int
    ) -> tuple[float, dict[str, Any]]:
        """Simulate neural network training with different fidelities."""
        lr = hyperparams["learning_rate"]
        wd = hyperparams["weight_decay"]
        batch_size = hyperparams["batch_size"]

        optimal_lr = 0.001
        optimal_wd = 0.0001

        lr_penalty = -5 * (np.log10(lr) - np.log10(optimal_lr)) ** 2
        wd_penalty = -2 * (np.log10(wd) - np.log10(optimal_wd)) ** 2
        batch_penalty = -0.0001 * (batch_size - 64) ** 2

        base_score = 0.8 + lr_penalty + wd_penalty + batch_penalty

        # Performance improves with training time (fidelity = epochs)
        if fidelity <= 20:
            fidelity_bonus = 0.15 * (1 - np.exp(-fidelity / 5))
        else:
            # Potential overfitting for very long training
            fidelity_bonus = 0.15 * (1 - np.exp(-20 / 5)) - 0.01 * (fidelity - 20)

        noise_std = 0.02 / np.sqrt(fidelity)
        noise = np.random.normal(0, noise_std)

        final_score = base_score + fidelity_bonus + noise

        training_time = fidelity * (1 + batch_size / 1000)

        metadata = {
            "base_score": base_score,
            "fidelity_bonus": fidelity_bonus,
            "training_time": training_time,
            "epochs_trained": fidelity,
        }

        return final_score, metadata

    nn_configs = []
    learning_rates = np.logspace(-5, -1, 5)  # 1e-5 to 1e-1
    weight_decays = np.logspace(-6, -2, 4)  # 1e-6 to 1e-2
    batch_sizes = [16, 32, 64, 128]

    for lr in learning_rates[:3]:
        for wd in weight_decays[:3]:
            for bs in batch_sizes[:2]:
                config = {"learning_rate": lr, "weight_decay": wd, "batch_size": bs}
                nn_configs.append(config)

    print(f"Generated {len(nn_configs)} neural network configurations")

    result2 = asha_optimize(
        eval_function=nn_eval,
        initial_configurations=nn_configs,
        min_fidelity=1,
        max_fidelity=81,
        reduction_factor=3,
        max_iterations=80,
        max_concurrent=4,
        random_state=42,
        verbose=True,
    )

    print(f"\nBest neural network configuration:")
    for param, value in result2.best_config.items():
        if isinstance(value, float):
            print(f"  {param}: {value:.2e}")
        else:
            print(f"  {param}: {value}")

    print(f"Best validation accuracy: {result2.best_score:.4f}")
    print(f"Training epochs used: {result2.best_fidelity}")
    print(f"Total budget used: {result2.total_budget_used} epoch-equivalents")

    correlations = analyze_fidelity_correlation(result2)
    if correlations:
        print(f"\nFidelity correlations:")
        for pair, corr in correlations.items():
            print(f"  {pair}: {corr:.3f}")

    print("\n" + "=" * 60)
    print("Example 3: Budget Efficiency Analysis")

    def random_search_baseline(eval_func, configs, max_budget, n_evals):
        """Simple random search baseline at maximum fidelity."""
        best_score = -np.inf
        best_config = None
        total_budget = 0

        sampled_configs = np.random.choice(
            len(configs), min(n_evals, len(configs)), replace=False
        )

        for i in sampled_configs:
            config = configs[i]
            score, metadata = eval_func(config, max_budget)
            total_budget += max_budget

            if score > best_score:
                best_score = score
                best_config = config

        return best_score, best_config, total_budget

    baseline_score, baseline_config, baseline_budget = random_search_baseline(
        nn_eval,
        nn_configs,
        81,
        10,  # 10 full evaluations
    )

    print(f"Random Search Baseline (10 full evaluations):")
    print(f"  Best score: {baseline_score:.4f}")
    print(f"  Total budget: {baseline_budget}")
    print(f"  Budget efficiency: {baseline_score / baseline_budget:.6f}")

    print(f"\nASHA Results:")
    print(f"  Best score: {result2.best_score:.4f}")
    print(f"  Total budget: {result2.total_budget_used}")
    print(f"  Budget efficiency: {result2.statistics['budget_efficiency']:.6f}")

    efficiency_improvement = result2.statistics["budget_efficiency"] / (
        baseline_score / baseline_budget
    )
    print(f"  Efficiency improvement: {efficiency_improvement:.2f}x")

    print(f"\nASHA completed {result2.statistics['total_evaluations']} evaluations")
    print(f"vs {10} evaluations for random search")
    print(
        f"ASHA found better solution with {efficiency_improvement:.1f}x better budget efficiency!"
    )
