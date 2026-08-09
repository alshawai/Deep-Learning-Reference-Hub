"""
Population-Based Training Tests
===============================

PBT couples two mechanisms that are easy to get subtly wrong together:
exploitation copies a good worker's weights and hyperparameters over a bad one,
and exploration then perturbs the copy. Break either and the run still completes,
still reports a best worker, and has become a population of independent random
searches.

The training functions here are pure functions of the hyperparameters, so the
worker that should win is known before the run starts and the tests can name it.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import numpy as np
import pytest
from conftest import load

pbt = load("hyperparameter_tuning/population_based_training.py")
WorkerState = pbt.WorkerState
PBTResult = pbt.PBTResult
HyperparameterDistribution = pbt.HyperparameterDistribution
LogUniformPerturbation = pbt.LogUniformPerturbation
UniformPerturbation = pbt.UniformPerturbation
ChoicePerturbation = pbt.ChoicePerturbation
WorkerInterface = pbt.WorkerInterface
FunctionWorker = pbt.FunctionWorker
PopulationBasedTrainer = pbt.PopulationBasedTrainer
pbt_optimize = pbt.pbt_optimize


BEST_LR = 0.01


def scored_train(hyperparams, steps):
    """Peaks at BEST_LR, so the winning configuration is known in advance."""
    return -abs(hyperparams["lr"] - BEST_LR), {"lr": hyperparams["lr"]}


class RecordingWorker(WorkerInterface):
    """A worker that remembers every call made to it."""

    def __init__(self, log):
        self.log = log
        self.state = None

    def train_step(self, hyperparams, steps=1):
        self.log.append(("train", dict(hyperparams), steps))
        return scored_train(hyperparams, steps)

    def save_state(self):
        self.log.append(("save",))
        return self.state

    def load_state(self, state):
        self.log.append(("load", state))
        self.state = state

    def reset(self):
        self.log.append(("reset",))


def trainer(**kwargs):
    options = {
        "worker_factory": lambda: FunctionWorker(
            scored_train, lambda: None, lambda state: None, lambda: None
        ),
        "initial_hyperparams": [{"lr": lr} for lr in (0.1, 0.05, 0.02, BEST_LR)],
        "hyperparam_distributions": {
            "lr": LogUniformPerturbation((0.9, 1.1), (1e-4, 1e-1))
        },
        "population_size": 4,
        "eval_interval": 10,
        "random_state": 0,
    }
    options.update(kwargs)
    return PopulationBasedTrainer(**options)


class TestPerturbations:
    def test_a_log_uniform_perturbation_scales_the_value(self):
        dist = LogUniformPerturbation(factor_range=(2.0, 2.0))
        assert dist.perturb(0.01) == pytest.approx(0.02)

    def test_a_log_uniform_perturbation_respects_its_bounds(self):
        dist = LogUniformPerturbation(factor_range=(100.0, 100.0), bounds=(1e-4, 1e-1))
        assert dist.perturb(0.01) == pytest.approx(0.1)

    def test_log_uniform_resampling_stays_inside_the_bounds(self):
        np.random.seed(0)
        dist = LogUniformPerturbation(bounds=(1e-5, 1e-1))
        draws = [dist.resample() for _ in range(200)]
        assert all(1e-5 <= d <= 1e-1 for d in draws)

    def test_log_uniform_resampling_spreads_across_the_orders_of_magnitude(self):
        """
        The reason to have a log-uniform distribution at all: a learning rate
        drawn uniformly from (1e-5, 1e-1) is almost never smaller than 1e-2.
        """
        np.random.seed(0)
        draws = [
            LogUniformPerturbation(bounds=(1e-5, 1e-1)).resample() for _ in range(500)
        ]
        decades = {int(np.floor(np.log10(d))) for d in draws}
        assert len(decades) >= 4

    def test_a_uniform_perturbation_adds_noise_around_the_value(self):
        np.random.seed(0)
        dist = UniformPerturbation(noise_std=0.1)
        draws = [dist.perturb(1.0) for _ in range(500)]
        assert np.mean(draws) == pytest.approx(1.0, abs=0.02)
        assert np.std(draws) == pytest.approx(0.1, abs=0.02)

    def test_a_uniform_perturbation_respects_its_bounds(self):
        np.random.seed(0)
        dist = UniformPerturbation(noise_std=10.0, bounds=(0.0, 1.0))
        assert all(0.0 <= dist.perturb(0.5) <= 1.0 for _ in range(100))

    @pytest.mark.parametrize(
        "distribution", [LogUniformPerturbation(), UniformPerturbation()]
    )
    def test_resampling_without_bounds_is_an_error(self, distribution):
        """There is no distribution to sample from without a range to sample over."""
        with pytest.raises(ValueError):
            distribution.resample()

    def test_a_choice_perturbation_only_ever_returns_a_listed_choice(self):
        np.random.seed(0)
        dist = ChoicePerturbation(["adam", "sgd", "rmsprop"], change_probability=1.0)
        assert {dist.perturb("adam") for _ in range(50)} <= {"sgd", "rmsprop"}

    def test_a_choice_perturbation_of_zero_probability_never_changes(self):
        dist = ChoicePerturbation(["adam", "sgd"], change_probability=0.0)
        assert all(dist.perturb("adam") == "adam" for _ in range(20))

    def test_a_single_choice_has_nothing_to_change_to(self):
        dist = ChoicePerturbation(["adam"], change_probability=1.0)
        assert dist.perturb("adam") == "adam"

    def test_the_abstract_distribution_refuses_to_be_instantiated(self):
        with pytest.raises(TypeError):
            HyperparameterDistribution()


class TestWorkerInterface:
    def test_the_abstract_worker_refuses_to_be_instantiated(self):
        with pytest.raises(TypeError):
            WorkerInterface()

    def test_the_function_worker_forwards_each_call_to_its_function(self):
        calls = []
        worker = FunctionWorker(
            lambda hp, steps: (calls.append(("train", hp, steps)) or 1.0, "state"),
            lambda: calls.append(("save",)) or "saved",
            lambda state: calls.append(("load", state)),
            lambda: calls.append(("reset",)),
        )
        assert worker.train_step({"lr": 0.1}, 5) == (1.0, "state")
        assert worker.save_state() == "saved"
        worker.load_state("s")
        worker.reset()
        assert calls == [
            ("train", {"lr": 0.1}, 5),
            ("save",),
            ("load", "s"),
            ("reset",),
        ]


class TestPopulation:
    def test_the_population_is_filled_to_size_by_resampling(self):
        """Two configurations were given; the other three have to come from somewhere."""
        t = trainer(initial_hyperparams=[{"lr": 0.1}, {"lr": 0.01}], population_size=5)
        t._initialize_population()
        assert len(t.population) == 5
        assert all(1e-4 <= w.hyperparams["lr"] <= 1e-1 for w in t.population)

    def test_every_worker_gets_its_own_hyperparameter_dictionary(self):
        """
        Sharing one dict would make a perturbation of any worker a perturbation
        of the whole population, which is exactly the diversity PBT exists to
        maintain.
        """
        t = trainer(population_size=4)
        t._initialize_population()
        assert all(
            a.hyperparams is not b.hyperparams
            for a in t.population
            for b in t.population
            if a is not b
        )

    def test_the_configurations_the_caller_passed_in_are_left_alone(self):
        """
        Exploration writes into a worker's hyperparameters in place. Those must
        not be the caller's own dictionaries, or a second run would start from
        wherever the first one drifted to.
        """
        given = [{"lr": 0.1}, {"lr": 0.05}, {"lr": 0.02}, {"lr": BEST_LR}]
        trainer(initial_hyperparams=given, population_size=4).train(
            max_steps=200, max_generations=5, verbose=False
        )
        assert given == [{"lr": 0.1}, {"lr": 0.05}, {"lr": 0.02}, {"lr": BEST_LR}]

    def test_worker_ids_are_distinct(self):
        t = trainer(population_size=6)
        t._initialize_population()
        assert sorted(w.worker_id for w in t.population) == list(range(6))

    def test_every_worker_is_reset_before_the_first_generation(self):
        log = []
        t = trainer(worker_factory=lambda: RecordingWorker(log), population_size=3)
        t.train(max_steps=30, max_generations=1, verbose=False)
        assert log[:3] == [("reset",)] * 3


class TestExploitAndExplore:
    def test_the_worst_worker_takes_on_the_best_worker_configuration(self):
        t = trainer(population_size=4, exploit_fraction=0.25)
        t._initialize_population()
        for worker, lr in zip(t.population, [0.1, 0.05, 0.02, BEST_LR]):
            worker.hyperparams = {"lr": lr}
        workers = [t.worker_factory() for _ in range(4)]

        t._exploit_and_explore(workers, [-0.09, -0.04, -0.01, 0.0])

        # The loser's learning rate was 0.1; it should now be a perturbation of
        # the winner's 0.01, not of its own.
        assert t.population[0].hyperparams["lr"] == pytest.approx(BEST_LR, rel=0.2)

    def test_exploitation_clears_the_history_of_the_worker_it_overwrites(self):
        """
        The scores belonged to a configuration that no longer exists. Keeping
        them would let a replaced worker be credited for a result it did not
        produce.
        """
        t = trainer(population_size=4, exploit_fraction=0.25)
        t._initialize_population()
        t.population[0].performance_history = [-5.0, -4.0]
        t._exploit_and_explore([t.worker_factory() for _ in range(4)], [-9, -4, -1, 0])
        assert t.population[0].performance_history == []

    def test_exploitation_copies_the_model_state_not_just_the_hyperparameters(self):
        """
        The point of PBT over repeated random search: a replaced worker inherits
        weights and continues from them rather than starting over.
        """
        log = []
        t = trainer(population_size=4, exploit_fraction=0.25)
        t._initialize_population()
        t.population[3].model_state = "winning-weights"
        workers = [RecordingWorker(log) for _ in range(4)]
        t._exploit_and_explore(workers, [-9.0, -4.0, -1.0, 0.0])
        assert ("load", "winning-weights") in log
        assert t.population[0].model_state == "winning-weights"

    def test_the_best_worker_is_left_alone(self):
        t = trainer(population_size=4, exploit_fraction=0.25)
        t._initialize_population()
        for worker, lr in zip(t.population, [0.1, 0.05, 0.02, BEST_LR]):
            worker.hyperparams = {"lr": lr}
        t._exploit_and_explore([t.worker_factory() for _ in range(4)], [-9, -4, -1, 0])
        assert t.population[3].hyperparams["lr"] == BEST_LR

    @pytest.mark.parametrize("fraction", [0.6, 0.9, 1.0])
    def test_a_large_exploit_fraction_never_overwrites_a_top_performer(self, fraction):
        """
        Regression test. The worst and best index sets are slices of one sorted
        array; above half the population they overlap, so a top performer landed
        in the list of workers to overwrite and was replaced and then perturbed
        away. The generation's best result was destroyed by the step meant to
        propagate it.
        """
        t = trainer(population_size=4, exploit_fraction=fraction)
        t._initialize_population()
        for worker, lr in zip(t.population, [0.1, 0.05, 0.02, BEST_LR]):
            worker.hyperparams = {"lr": lr}
        t._exploit_and_explore([t.worker_factory() for _ in range(4)], [-9, -4, -1, 0])
        assert t.population[3].hyperparams["lr"] == BEST_LR

    def test_a_population_too_small_to_rank_is_left_untouched(self):
        t = trainer(population_size=1, initial_hyperparams=[{"lr": 0.1}])
        t._initialize_population()
        t._exploit_and_explore([t.worker_factory()], [-1.0])
        assert t.population[0].hyperparams == {"lr": 0.1}

    def test_perturbation_moves_a_hyperparameter_without_leaving_its_bounds(self):
        t = trainer(population_size=2)
        t._initialize_population()
        t.population[0].hyperparams = {"lr": 0.05}
        t._perturb_hyperparams(0)
        assert t.population[0].hyperparams["lr"] != 0.05
        assert 1e-4 <= t.population[0].hyperparams["lr"] <= 1e-1

    def test_a_hyperparameter_without_a_distribution_is_left_as_it_was(self):
        """
        Not every entry in a configuration is something to search over; a fixed
        batch size must survive the run unchanged.
        """
        t = trainer(
            initial_hyperparams=[{"lr": 0.05, "batch_size": 32}],
            population_size=2,
        )
        t._initialize_population()
        for _ in range(20):
            t._perturb_hyperparams(0)
        assert t.population[0].hyperparams["batch_size"] == 32


class TestTrain:
    def test_the_population_converges_on_the_configuration_that_scores_best(self):
        """
        The claim PBT makes. Started with three poor learning rates and one good
        one, the population should end up near the good one rather than spread
        over its starting range.
        """
        result = trainer(population_size=8, exploit_fraction=0.25).train(
            max_steps=8 * 10 * 15, max_generations=15, verbose=False
        )
        final = [w.hyperparams["lr"] for w in result.final_population]
        assert np.median(final) == pytest.approx(BEST_LR, abs=0.01)

    def test_the_best_worker_is_the_best_score_seen_anywhere(self):
        result = trainer(population_size=4).train(
            max_steps=200, max_generations=5, verbose=False
        )
        every_score = [
            score
            for generation in result.population_history
            for worker in generation
            for score in worker.performance_history
        ]
        assert max(result.best_worker.performance_history) == max(every_score)

    def test_a_worker_that_diverges_does_not_become_the_reported_best(self):
        """
        Regression test. np.argsort places nan last, so a worker whose loss went
        to nan was read as the population's best: its hyperparameters were
        copied over everyone else by exploitation and returned as the answer.
        """

        def diverges_above(hyperparams, steps):
            if hyperparams["lr"] > 0.05:
                return float("nan"), None
            return scored_train(hyperparams, steps)

        result = pbt_optimize(
            diverges_above,
            lambda: None,
            lambda state: None,
            lambda: None,
            [{"lr": lr} for lr in (0.09, 0.08, 0.02, BEST_LR)],
            {"lr": LogUniformPerturbation((0.9, 1.1), (1e-4, 5e-2))},
            population_size=4,
            max_steps=400,
            eval_interval=10,
            random_state=0,
            verbose=False,
        )
        assert result.best_worker.hyperparams["lr"] <= 0.05
        assert np.isfinite(max(result.best_worker.performance_history))

    def test_a_worker_that_raises_is_recorded_as_the_worst_score(self):
        def explodes_above(hyperparams, steps):
            if hyperparams["lr"] > 0.05:
                raise RuntimeError("simulated divergence")
            return scored_train(hyperparams, steps)

        with pytest.warns(UserWarning, match="evaluation failed"):
            result = pbt_optimize(
                explodes_above,
                lambda: None,
                lambda state: None,
                lambda: None,
                [{"lr": lr} for lr in (0.09, 0.08, 0.02, BEST_LR)],
                {"lr": LogUniformPerturbation((0.9, 1.1), (1e-4, 5e-2))},
                population_size=4,
                max_steps=200,
                eval_interval=10,
                random_state=0,
                verbose=False,
            )
        assert np.isfinite(max(result.best_worker.performance_history))

    def test_max_generations_bounds_the_run(self):
        result = trainer(population_size=4).train(
            max_steps=10**9, max_generations=3, verbose=False
        )
        assert result.statistics["generations_completed"] == 3
        assert len(result.population_history) == 3

    def test_max_steps_bounds_the_run(self):
        result = trainer(population_size=4, eval_interval=10).train(
            max_steps=80, max_generations=10**6, verbose=False
        )
        assert result.total_steps == 80

    def test_the_step_count_is_the_whole_population_not_one_worker(self):
        result = trainer(population_size=4, eval_interval=10).train(
            max_steps=120, max_generations=3, verbose=False
        )
        assert result.total_steps == 3 * 4 * 10

    def test_each_generation_is_snapshotted_rather_than_aliased(self):
        """
        The history is used to plot how hyperparameters moved. Storing live
        references would make every generation show the final values.
        """
        result = trainer(population_size=4).train(
            max_steps=200, max_generations=5, verbose=False
        )
        first = result.population_history[0][0]
        assert first is not result.final_population[0]
        assert first.hyperparams is not result.final_population[0].hyperparams

    def test_the_random_state_makes_a_run_reproducible(self):
        first = trainer(random_state=7).train(
            max_steps=200, max_generations=5, verbose=False
        )
        second = trainer(random_state=7).train(
            max_steps=200, max_generations=5, verbose=False
        )
        assert (
            first.statistics["best_score_progression"]
            == (second.statistics["best_score_progression"])
        )

    def test_verbose_false_prints_nothing(self, capsys):
        trainer(population_size=4).train(
            max_steps=200, max_generations=5, verbose=False
        )
        assert capsys.readouterr().out == ""


class TestStatistics:
    def test_the_best_score_never_goes_backwards(self):
        result = trainer(population_size=6).train(
            max_steps=600, max_generations=10, verbose=False
        )
        progression = result.statistics["best_score_progression"]
        assert progression == sorted(progression)

    def test_diversity_falls_as_the_population_converges(self):
        """
        The trade PBT makes: exploitation narrows the population onto what works,
        at the cost of the spread that let it discover anything.
        """
        result = trainer(population_size=8, exploit_fraction=0.25).train(
            max_steps=8 * 10 * 15, max_generations=15, verbose=False
        )
        diversity = result.statistics["population_diversity"]
        assert diversity["final_diversity"] < diversity["initial_diversity"]
        assert diversity["diversity_trend"] < 0

    def test_diversity_is_empty_before_any_generation_has_run(self):
        assert trainer()._compute_diversity_metrics() == {}

    def test_a_run_too_short_to_judge_reports_no_convergence_generation(self):
        result = trainer(population_size=4).train(
            max_steps=160, max_generations=4, verbose=False
        )
        assert result.statistics["convergence_generation"] is None

    def test_a_plateau_is_reported_as_the_generation_it_started(self):
        flat = lambda hyperparams, steps: (1.0, None)
        result = pbt_optimize(
            flat,
            lambda: None,
            lambda state: None,
            lambda: None,
            [{"lr": 0.01}],
            {"lr": LogUniformPerturbation((0.9, 1.1), (1e-4, 1e-1))},
            population_size=4,
            max_steps=400,
            eval_interval=10,
            random_state=0,
            verbose=False,
        )
        assert result.statistics["convergence_generation"] == 5

    def test_the_result_is_the_documented_dataclass(self):
        result = trainer(population_size=4).train(
            max_steps=200, max_generations=5, verbose=False
        )
        assert isinstance(result, PBTResult)
        assert isinstance(result.best_worker, WorkerState)
        assert len(result.final_population) == 4
        assert result.total_training_time > 0
