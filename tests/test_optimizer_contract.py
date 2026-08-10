"""
Optimizer Contract Tests
========================

The base class carries no update rule, so nothing here checks mathematics. What
it checks is the contract every optimizer in the subpackage is held to, which is
what a driver holding an unknown optimizer is entitled to assume.

Three properties, each of which fails silently when broken. An unimplemented
update that returned the parameters unchanged would produce a flat loss curve
that reads as instant convergence. A reset that left state behind would make the
second run in a session differ from the first, which is invisible unless someone
runs the same comparison twice. An update that mutated the caller's dictionary
would leave every recorded step of a trajectory pointing at the final values, so
the plot of the descent would show a flat line at the answer.

The concrete implementations exercised here are the comparison harness's, since
they are the ones written to this contract.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import numpy as np
import pytest

from dlhub.optimizers.base import BaseOptimizer
from dlhub.optimizers.comparison import (
    AdamOptimizer,
    MomentumOptimizer,
    RMSpropOptimizer,
    SGDOptimizer,
)

STATEFUL = [MomentumOptimizer, RMSpropOptimizer, AdamOptimizer]
STATEFUL_IDS = ["momentum", "rmsprop", "adam"]

EVERY_OPTIMIZER = [SGDOptimizer, *STATEFUL]
EVERY_OPTIMIZER_IDS = ["sgd", *STATEFUL_IDS]


@pytest.fixture
def params():
    return {"W": np.array([[1.0, -2.0], [0.5, 3.0]]), "b": np.array([0.25, -0.75])}


@pytest.fixture
def grads():
    return {"W": np.array([[0.1, 0.2], [-0.3, 0.4]]), "b": np.array([-0.05, 0.15])}


class TestTheUpdateMustBeImplemented:
    def test_the_base_class_refuses_to_update(self, params, grads):
        """
        An optimizer is its update rule. Inheriting a no-op default would give a
        subclass that forgot to implement one a loss curve flat at its starting
        value, which reads as convergence at iteration one.
        """
        with pytest.raises(NotImplementedError):
            BaseOptimizer().update_parameters(params, grads, 1)

    def test_a_subclass_that_does_not_override_it_still_refuses(self, params, grads):
        class Forgetful(BaseOptimizer):
            """Sets a name, implements nothing."""

            def __init__(self):
                super().__init__()
                self.name = "Forgetful"

        with pytest.raises(NotImplementedError):
            Forgetful().update_parameters(params, grads, 1)

    @pytest.mark.parametrize(
        "optimizer_class", EVERY_OPTIMIZER, ids=EVERY_OPTIMIZER_IDS
    )
    def test_every_shipped_optimizer_implements_it(
        self, optimizer_class, params, grads
    ):
        updated = optimizer_class().update_parameters(params, grads, 1)
        assert set(updated) == set(params)


class TestResetClearsAccumulatedState:
    @pytest.mark.parametrize("optimizer_class", STATEFUL, ids=STATEFUL_IDS)
    def test_a_reset_optimizer_repeats_its_first_run_exactly(
        self, optimizer_class, params, grads
    ):
        """
        The observable form of "reset clears state": the trajectory of a run is a
        function of the run, not of what the optimizer did before it. Asserting
        against the state attributes instead would pass for an optimizer that
        zeroed them while leaving a step counter set.
        """

        def run(optimizer):
            current = dict(params)
            trajectory = []
            for t in range(1, 6):
                current = optimizer.update_parameters(current, grads, t)
                trajectory.append(current["W"].copy())
            return trajectory

        optimizer = optimizer_class()
        first = run(optimizer)
        optimizer.reset()
        second = run(optimizer)

        for step, (before, after) in enumerate(zip(first, second), start=1):
            assert np.allclose(before, after), f"step {step} differs after reset"

    @pytest.mark.parametrize("optimizer_class", STATEFUL, ids=STATEFUL_IDS)
    def test_state_actually_accumulates_between_steps(
        self, optimizer_class, params, grads
    ):
        """
        Guards the test above from passing vacuously: an optimizer holding no
        state at all would repeat its run whether or not reset did anything.
        """
        optimizer = optimizer_class()
        first_step = optimizer.update_parameters(params, grads, 1)
        second_step = optimizer.update_parameters(params, grads, 2)
        assert not np.allclose(
            first_step["W"] - params["W"], second_step["W"] - first_step["W"]
        )

    def test_resetting_a_stateless_optimizer_is_harmless(self, params, grads):
        """Plain gradient descent has nothing to clear, and inherits the no-op."""
        optimizer = SGDOptimizer()
        before = optimizer.update_parameters(params, grads, 1)
        optimizer.reset()
        assert np.allclose(
            optimizer.update_parameters(params, grads, 1)["W"], before["W"]
        )


class TestTheCallersParametersAreLeftAlone:
    @pytest.mark.parametrize(
        "optimizer_class", EVERY_OPTIMIZER, ids=EVERY_OPTIMIZER_IDS
    )
    def test_an_update_returns_a_new_dictionary(self, optimizer_class, params, grads):
        updated = optimizer_class().update_parameters(params, grads, 1)
        assert updated is not params
        for key in params:
            assert updated[key] is not params[key]

    @pytest.mark.parametrize(
        "optimizer_class", EVERY_OPTIMIZER, ids=EVERY_OPTIMIZER_IDS
    )
    def test_an_update_does_not_write_into_the_arrays_it_was_given(
        self, optimizer_class, params, grads
    ):
        """
        A run records ``params.copy()`` at each iteration, which copies the dict
        but not the arrays inside it. An in-place update would leave every
        recorded step aliasing the final parameters.
        """
        original = {key: value.copy() for key, value in params.items()}
        optimizer_class().update_parameters(params, grads, 1)
        for key, value in original.items():
            assert np.array_equal(params[key], value)

    @pytest.mark.parametrize(
        "optimizer_class", EVERY_OPTIMIZER, ids=EVERY_OPTIMIZER_IDS
    )
    def test_an_update_does_not_write_into_the_gradients(
        self, optimizer_class, params, grads
    ):
        original = {key: value.copy() for key, value in grads.items()}
        optimizer_class().update_parameters(params, grads, 1)
        for key, value in original.items():
            assert np.array_equal(grads[key], value)


class TestEveryOptimizerNamesItself:
    @pytest.mark.parametrize(
        "optimizer_class", EVERY_OPTIMIZER, ids=EVERY_OPTIMIZER_IDS
    )
    def test_the_name_says_which_method_it_is(self, optimizer_class):
        """
        Results are keyed and plotted by name, so an optimizer inheriting the
        base label would collide with every other one that forgot to set it.
        """
        name = optimizer_class().name
        assert name and name != BaseOptimizer().name
