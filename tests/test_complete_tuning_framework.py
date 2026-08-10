"""
Complete Tuning Framework Tests
===============================

This module is the front door to the chapter: the sampler, the trial bookkeeping,
the logger, and the two search strategies that sit on top of them. Most of what
can go wrong here is silent -- a sampler that ignores its log scale still returns
learning rates, a `maximize=False` run still reports a best trial, and a logger
that writes nothing still returns a result object.

The objectives below are pure functions with a known optimum, so the trial that
should win is known before the run starts.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import json

import numpy as np
import pytest

from dlhub.tuning.framework import (
    ExperimentConfig,
    ExperimentLogger,
    FunctionObjective,
    HyperparameterConfig,
    HyperparameterOptimizer,
    HyperparameterSampler,
    ObjectiveFunction,
    OptimizationMethod,
    OptimizationResult,
    TrialResult,
    optimize_hyperparameters,
)

BEST_LR = 0.01

SPACE = [
    {"name": "lr", "type": "continuous", "range": (1e-4, 1e-1), "scale": "log"},
    {"name": "layers", "type": "integer", "range": (1, 4)},
    {"name": "optimizer", "type": "categorical", "range": ["adam", "sgd"]},
]


def accuracy(hyperparams):
    """Peaks at BEST_LR, so the winning trial is known in advance."""
    score = -abs(hyperparams["lr"] - BEST_LR)
    return {"accuracy": score, "loss": -score}


def optimizer_for(configs, method=OptimizationMethod.GRID_SEARCH, **kwargs):
    options = {
        "experiment_name": "test",
        "optimization_method": method,
        "hyperparameters": configs,
        "objective_metric": "accuracy",
        "n_trials": 100,
    }
    options.update(kwargs)
    return HyperparameterOptimizer(
        ExperimentConfig(**options), FunctionObjective(accuracy, ["accuracy", "loss"])
    )


class TestSampler:
    def test_a_sample_covers_every_declared_hyperparameter(self):
        configs = [
            HyperparameterConfig("lr", "continuous", (1e-4, 1e-1), "log"),
            HyperparameterConfig("layers", "integer", (1, 4)),
            HyperparameterConfig("optimizer", "categorical", ["adam", "sgd"]),
        ]
        sample = HyperparameterSampler(configs, random_state=0).sample()
        assert set(sample) == {"lr", "layers", "optimizer"}

    def test_continuous_samples_stay_inside_the_range(self):
        config = HyperparameterConfig("lr", "continuous", (0.2, 0.8))
        sampler = HyperparameterSampler([config], random_state=0)
        assert all(0.2 <= sampler.sample()["lr"] <= 0.8 for _ in range(200))

    def test_a_log_scale_spreads_the_samples_across_the_decades(self):
        """
        The reason the scale is configurable. Sampled uniformly, 90% of the draws
        from (1e-5, 1e-1) land in the top decade and the small learning rates are
        never tried at all.
        """
        config = HyperparameterConfig("lr", "continuous", (1e-5, 1e-1), "log")
        sampler = HyperparameterSampler([config], random_state=0)
        draws = [sampler.sample()["lr"] for _ in range(400)]
        decades = {int(np.floor(np.log10(d))) for d in draws}
        assert len(decades) >= 4
        assert np.median(draws) < 1e-2

    def test_a_linear_scale_keeps_the_samples_uniform(self):
        config = HyperparameterConfig("lr", "continuous", (0.0, 1.0))
        sampler = HyperparameterSampler([config], random_state=0)
        draws = [sampler.sample()["lr"] for _ in range(500)]
        assert np.mean(draws) == pytest.approx(0.5, abs=0.05)

    def test_integer_samples_are_integers_within_the_inclusive_range(self):
        """
        Both endpoints must be reachable: a range of (1, 4) that never returns 4
        quietly shrinks the search space.
        """
        config = HyperparameterConfig("layers", "integer", (1, 4))
        sampler = HyperparameterSampler([config], random_state=0)
        draws = {sampler.sample()["layers"] for _ in range(300)}
        assert draws == {1, 2, 3, 4}
        assert all(isinstance(d, (int, np.integer)) for d in draws)

    def test_categorical_samples_only_ever_come_from_the_choices(self):
        config = HyperparameterConfig(
            "optimizer", "categorical", ["adam", "sgd", "rms"]
        )
        sampler = HyperparameterSampler([config], random_state=0)
        assert {sampler.sample()["optimizer"] for _ in range(200)} == {
            "adam",
            "sgd",
            "rms",
        }

    def test_an_unknown_parameter_type_is_rejected(self):
        config = HyperparameterConfig("lr", "quantum", (0.0, 1.0))
        with pytest.raises(ValueError, match="Unknown parameter type"):
            HyperparameterSampler([config]).sample()

    def test_the_seed_makes_sampling_reproducible(self):
        configs = [HyperparameterConfig("lr", "continuous", (1e-4, 1e-1), "log")]
        first = [HyperparameterSampler(configs, 5).sample()["lr"] for _ in range(3)]
        second = [HyperparameterSampler(configs, 5).sample()["lr"] for _ in range(3)]
        assert first[0] == second[0]


class TestValidation:
    def test_a_sample_from_the_space_validates(self):
        configs = [
            HyperparameterConfig("lr", "continuous", (1e-4, 1e-1), "log"),
            HyperparameterConfig("optimizer", "categorical", ["adam", "sgd"]),
        ]
        sampler = HyperparameterSampler(configs, random_state=0)
        assert sampler.validate(sampler.sample())

    def test_a_missing_hyperparameter_fails_validation(self):
        configs = [HyperparameterConfig("lr", "continuous", (0.0, 1.0))]
        assert not HyperparameterSampler(configs).validate({})

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_a_value_outside_the_range_fails_validation(self, value):
        configs = [HyperparameterConfig("lr", "continuous", (0.0, 1.0))]
        assert not HyperparameterSampler(configs).validate({"lr": value})

    def test_an_unlisted_category_fails_validation(self):
        configs = [HyperparameterConfig("opt", "categorical", ["adam", "sgd"])]
        assert not HyperparameterSampler(configs).validate({"opt": "lbfgs"})

    def test_the_range_endpoints_are_valid(self):
        configs = [HyperparameterConfig("lr", "continuous", (0.0, 1.0))]
        sampler = HyperparameterSampler(configs)
        assert sampler.validate({"lr": 0.0}) and sampler.validate({"lr": 1.0})


class TestObjective:
    def test_the_abstract_objective_refuses_to_be_instantiated(self):
        with pytest.raises(TypeError):
            ObjectiveFunction()

    def test_the_function_objective_forwards_the_call_and_reports_its_metrics(self):
        seen = []
        objective = FunctionObjective(
            lambda hp: seen.append(hp) or {"accuracy": 1.0}, ["accuracy"]
        )
        assert objective.evaluate({"lr": 0.1}) == {"accuracy": 1.0}
        assert objective.get_metric_names() == ["accuracy"]
        assert seen == [{"lr": 0.1}]


class TestTrials:
    def test_a_trial_records_the_metrics_and_a_success_status(self):
        opt = optimizer_for([HyperparameterConfig("lr", "continuous", (1e-4, 1e-1))])
        trial = opt._evaluate_trial({"lr": BEST_LR})
        assert trial.status == "success"
        assert trial.metrics["accuracy"] == pytest.approx(0.0)
        assert trial.training_time >= 0.0

    def test_trial_ids_are_distinct_and_sequential(self):
        opt = optimizer_for([HyperparameterConfig("lr", "continuous", (1e-4, 1e-1))])
        ids = [opt._evaluate_trial({"lr": 0.01}).trial_id for _ in range(4)]
        assert ids == [0, 1, 2, 3]

    def test_a_trial_stores_its_own_copy_of_the_hyperparameters(self):
        """
        The caller's dict is reused across trials by some search strategies, so
        keeping a reference would make every recorded trial show the last values.
        """
        opt = optimizer_for([HyperparameterConfig("lr", "continuous", (1e-4, 1e-1))])
        hyperparams = {"lr": BEST_LR}
        trial = opt._evaluate_trial(hyperparams)
        hyperparams["lr"] = 999.0
        assert trial.hyperparams["lr"] == BEST_LR

    def test_an_objective_that_raises_is_recorded_as_a_failed_trial(self):
        opt = HyperparameterOptimizer(
            ExperimentConfig(
                "test",
                OptimizationMethod.RANDOM_SEARCH,
                [HyperparameterConfig("lr", "continuous", (1e-4, 1e-1))],
                "accuracy",
                n_trials=1,
            ),
            FunctionObjective(
                lambda hp: (_ for _ in ()).throw(RuntimeError("diverged")), ["accuracy"]
            ),
        )
        with pytest.warns(UserWarning, match="failed"):
            trial = opt._evaluate_trial({"lr": 0.05})
        assert trial.status == "failed"
        assert "diverged" in trial.metadata["error"]

    def test_a_configuration_outside_the_space_is_a_failed_trial(self):
        opt = optimizer_for([HyperparameterConfig("lr", "continuous", (1e-4, 1e-1))])
        with pytest.warns(UserWarning, match="Invalid hyperparameter"):
            trial = opt._evaluate_trial({"lr": 50.0})
        assert trial.status == "failed"

    def test_an_objective_that_omits_the_target_metric_is_a_failed_trial(self):
        """
        Otherwise the missing key surfaces as a KeyError while ranking trials,
        long after the run that could have reported it.
        """
        opt = HyperparameterOptimizer(
            ExperimentConfig(
                "test",
                OptimizationMethod.RANDOM_SEARCH,
                [HyperparameterConfig("lr", "continuous", (1e-4, 1e-1))],
                "accuracy",
                n_trials=1,
            ),
            FunctionObjective(lambda hp: {"loss": 0.5}, ["loss"]),
        )
        with pytest.warns(UserWarning, match="not found"):
            trial = opt._evaluate_trial({"lr": 0.05})
        assert trial.status == "failed"

    def test_a_failed_trial_never_becomes_the_best_trial(self):
        """
        A failure is scored -inf under maximization, but +inf under minimization
        -- where a plain comparison would rank it first.
        """
        for maximize in (True, False):
            opt = HyperparameterOptimizer(
                ExperimentConfig(
                    "test",
                    OptimizationMethod.RANDOM_SEARCH,
                    [HyperparameterConfig("lr", "continuous", (1e-4, 1e-1))],
                    "accuracy",
                    maximize=maximize,
                    n_trials=2,
                ),
                FunctionObjective(accuracy, ["accuracy", "loss"]),
            )
            opt._evaluate_trial({"lr": BEST_LR})
            with pytest.warns(UserWarning):
                opt._evaluate_trial({"lr": 50.0})
            assert opt.best_trial.status == "success"


class TestGrid:
    def test_the_grid_is_the_product_of_the_per_parameter_points(self):
        """Five points for the continuous axis, two categories: ten cells."""
        opt = optimizer_for(
            [
                HyperparameterConfig("lr", "continuous", (1e-4, 1e-1), "log"),
                HyperparameterConfig("optimizer", "categorical", ["adam", "sgd"]),
            ]
        )
        grid = opt._generate_grid()
        assert len(grid) == 10
        assert len({tuple(sorted(c.items())) for c in grid}) == 10

    def test_a_log_scaled_axis_is_spaced_geometrically(self):
        opt = optimizer_for(
            [HyperparameterConfig("lr", "continuous", (1e-4, 1e-1), "log")]
        )
        points = sorted(c["lr"] for c in opt._generate_grid())
        ratios = [b / a for a, b in zip(points, points[1:])]
        assert all(r == pytest.approx(ratios[0]) for r in ratios)

    def test_a_linear_axis_is_spaced_evenly(self):
        opt = optimizer_for([HyperparameterConfig("lr", "continuous", (0.0, 1.0))])
        points = sorted(c["lr"] for c in opt._generate_grid())
        assert points == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])

    def test_an_integer_axis_narrower_than_the_grid_is_not_repeated(self):
        """
        Five points over the integers 1..4 must round to four cells, not five
        with one evaluated twice.
        """
        opt = optimizer_for([HyperparameterConfig("n", "integer", (1, 4))])
        assert [c["n"] for c in opt._generate_grid()] == [1, 2, 3, 4]

    def test_the_grid_covers_both_endpoints(self):
        opt = optimizer_for(
            [HyperparameterConfig("lr", "continuous", (1e-4, 1e-1), "log")]
        )
        points = sorted(c["lr"] for c in opt._generate_grid())
        assert points[0] == pytest.approx(1e-4)
        assert points[-1] == pytest.approx(1e-1)

    def test_the_trial_budget_truncates_a_grid_larger_than_it(self):
        opt = optimizer_for(
            [HyperparameterConfig("lr", "continuous", (0.0, 1.0))], n_trials=3
        )
        opt.optimize(verbose=False)
        assert len(opt.all_trials) == 3


class TestOptimize:
    def test_random_search_finds_the_optimum_it_is_pointed_at(self):
        result = optimize_hyperparameters(
            accuracy, SPACE, "accuracy", n_trials=60, random_seed=0, verbose=False
        )
        assert result.best_trial.hyperparams["lr"] == pytest.approx(BEST_LR, abs=0.005)

    def test_the_best_trial_is_the_best_of_the_recorded_trials(self):
        result = optimize_hyperparameters(
            accuracy, SPACE, "accuracy", n_trials=20, random_seed=0, verbose=False
        )
        assert result.best_trial.metrics["accuracy"] == max(
            t.metrics["accuracy"] for t in result.all_trials
        )

    def test_minimizing_picks_the_lowest_rather_than_the_highest(self):
        """
        A sign error here is invisible in the shape of the result: it still
        returns a trial, still reports a metric, and is simply the worst one.
        """
        result = optimize_hyperparameters(
            accuracy,
            SPACE,
            "loss",
            n_trials=20,
            maximize=False,
            random_seed=0,
            verbose=False,
        )
        assert result.best_trial.metrics["loss"] == min(
            t.metrics["loss"] for t in result.all_trials
        )

    def test_the_run_spends_its_whole_trial_budget(self):
        result = optimize_hyperparameters(
            accuracy, SPACE, "accuracy", n_trials=15, random_seed=0, verbose=False
        )
        assert len(result.all_trials) == 15

    def test_every_trial_stays_inside_the_declared_space(self):
        result = optimize_hyperparameters(
            accuracy, SPACE, "accuracy", n_trials=30, random_seed=0, verbose=False
        )
        for trial in result.all_trials:
            assert 1e-4 <= trial.hyperparams["lr"] <= 1e-1
            assert trial.hyperparams["layers"] in {1, 2, 3, 4}
            assert trial.hyperparams["optimizer"] in {"adam", "sgd"}

    def test_grid_search_evaluates_each_cell_once(self):
        space = [
            {"name": "lr", "type": "continuous", "range": (1e-4, 1e-1), "scale": "log"},
            {"name": "optimizer", "type": "categorical", "range": ["adam", "sgd"]},
        ]
        result = optimize_hyperparameters(
            accuracy,
            space,
            "accuracy",
            optimization_method="grid_search",
            n_trials=1000,
            random_seed=0,
            verbose=False,
        )
        seen = {
            (t.hyperparams["lr"], t.hyperparams["optimizer"]) for t in result.all_trials
        }
        assert len(result.all_trials) == 10 == len(seen)

    def test_an_unimplemented_method_says_so_rather_than_returning_nothing(self):
        opt = optimizer_for(
            [HyperparameterConfig("lr", "continuous", (1e-4, 1e-1))],
            method=OptimizationMethod.PBT,
        )
        with pytest.raises(NotImplementedError):
            opt.optimize(verbose=False)

    def test_an_unknown_method_name_is_rejected(self):
        with pytest.raises(ValueError):
            optimize_hyperparameters(
                accuracy,
                SPACE,
                "accuracy",
                optimization_method="telepathy",
                n_trials=2,
                verbose=False,
            )

    def test_the_seed_makes_a_run_reproducible(self):
        def run():
            return optimize_hyperparameters(
                accuracy, SPACE, "accuracy", n_trials=12, random_seed=9, verbose=False
            )

        first, second = run(), run()
        assert [t.hyperparams["lr"] for t in first.all_trials] == [
            t.hyperparams["lr"] for t in second.all_trials
        ]

    def test_verbose_false_prints_nothing(self, capsys):
        optimize_hyperparameters(
            accuracy, SPACE, "accuracy", n_trials=10, random_seed=0, verbose=False
        )
        assert capsys.readouterr().out == ""

    def test_the_result_is_the_documented_dataclass(self):
        result = optimize_hyperparameters(
            accuracy, SPACE, "accuracy", n_trials=10, random_seed=0, verbose=False
        )
        assert isinstance(result, OptimizationResult)
        assert isinstance(result.best_trial, TrialResult)
        assert result.experiment_config.objective_metric == "accuracy"
        assert result.total_time > 0


class TestSummaryStatistics:
    def test_the_counts_add_up(self):
        result = optimize_hyperparameters(
            accuracy, SPACE, "accuracy", n_trials=12, random_seed=0, verbose=False
        )
        stats = result.summary_statistics
        assert stats["n_successful"] + stats["n_failed"] == stats["n_trials"] == 12

    def test_the_objective_statistics_describe_the_successful_trials(self):
        result = optimize_hyperparameters(
            accuracy, SPACE, "accuracy", n_trials=12, random_seed=0, verbose=False
        )
        scores = [
            t.metrics["accuracy"] for t in result.all_trials if t.status == "success"
        ]
        objective = result.summary_statistics["objective_statistics"]
        assert objective["max"] == pytest.approx(max(scores))
        assert objective["mean"] == pytest.approx(np.mean(scores))
        assert objective["median"] == pytest.approx(np.median(scores))

    def test_the_improvement_is_measured_against_the_first_ten_trials(self):
        result = optimize_hyperparameters(
            accuracy, SPACE, "accuracy", n_trials=20, random_seed=0, verbose=False
        )
        scores = [
            t.metrics["accuracy"] for t in result.all_trials if t.status == "success"
        ]
        baseline = np.mean(scores[:10])
        expected = (result.best_trial.metrics["accuracy"] - baseline) / abs(baseline)
        assert result.summary_statistics["improvement_over_random"] == pytest.approx(
            expected
        )

    def test_too_few_trials_to_form_a_baseline_reports_no_improvement(self):
        result = optimize_hyperparameters(
            accuracy, SPACE, "accuracy", n_trials=5, random_seed=0, verbose=False
        )
        assert result.summary_statistics["improvement_over_random"] == 0.0

    def test_a_metric_centred_on_zero_does_not_divide_by_zero(self):
        """
        Regression test. The improvement is a ratio against the baseline mean,
        and a metric that averages to exactly zero -- a centred score, a
        difference against a reference model -- made it nan with a runtime
        warning.
        """
        with np.errstate(all="raise"):
            result = optimize_hyperparameters(
                lambda hp: {"accuracy": 0.0},
                SPACE,
                "accuracy",
                n_trials=12,
                random_seed=0,
                verbose=False,
            )
        assert result.summary_statistics["improvement_over_random"] == 0.0

    def test_a_run_with_no_successful_trial_reports_no_statistics(self):
        opt = optimizer_for([HyperparameterConfig("lr", "continuous", (1e-4, 1e-1))])
        assert opt._compute_summary_statistics() == {}


class TestLogger:
    def test_a_logger_without_a_directory_writes_nothing(self, tmp_path):
        logger = ExperimentLogger(None, "test")
        logger.log_trial(TrialResult(0, {"lr": 0.1}, {"accuracy": 0.9}, 0.1))
        assert list(tmp_path.iterdir()) == []
        assert logger.load_results() is None

    def test_each_trial_is_appended_as_one_json_line(self, tmp_path):
        logger = ExperimentLogger(str(tmp_path), "test")
        for i in range(3):
            logger.log_trial(TrialResult(i, {"lr": 0.1 * i}, {"accuracy": 0.9}, 0.1))
        lines = (tmp_path / "test_log.jsonl").read_text().strip().split("\n")
        assert len(lines) == 3
        assert [json.loads(line)["trial_id"] for line in lines] == [0, 1, 2]

    def test_the_log_directory_is_created_if_it_does_not_exist(self, tmp_path):
        target = tmp_path / "nested" / "runs"
        ExperimentLogger(str(target), "test")
        assert target.is_dir()

    def test_the_summary_records_the_best_trial_and_the_method(self, tmp_path):
        optimize_hyperparameters(
            accuracy,
            SPACE,
            "accuracy",
            experiment_name="run",
            n_trials=12,
            random_seed=0,
            save_dir=str(tmp_path),
            verbose=False,
        )
        summary = json.loads((tmp_path / "run_summary.json").read_text())
        assert summary["optimization_method"] == "random_search"
        assert summary["n_trials"] == 12
        assert set(summary["best_trial"]) == {"hyperparams", "metrics", "trial_id"}

    def test_a_completed_run_can_be_read_back_from_its_log(self, tmp_path):
        """
        The log is the durable record; a run that cannot be reloaded from it has
        to be repeated to be analysed.
        """
        result = optimize_hyperparameters(
            accuracy,
            SPACE,
            "accuracy",
            experiment_name="run",
            n_trials=10,
            random_seed=0,
            save_dir=str(tmp_path),
            verbose=False,
        )
        reloaded = ExperimentLogger(str(tmp_path), "run").load_results()
        assert len(reloaded) == len(result.all_trials)
        assert all(isinstance(trial, TrialResult) for trial in reloaded)
        assert [t.trial_id for t in reloaded] == [t.trial_id for t in result.all_trials]
        assert reloaded[0].metrics["accuracy"] == pytest.approx(
            result.all_trials[0].metrics["accuracy"]
        )


def test_every_method_in_the_enum_has_a_distinct_value():
    values = [method.value for method in OptimizationMethod]
    assert len(values) == len(set(values))
