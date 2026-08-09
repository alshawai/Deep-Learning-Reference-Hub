"""
Gradient Checking Utility Tests
===============================

This module is the tool the rest of the hub uses to prove its backpropagation
correct, so its own correctness has to be established independently. A gradient
checker that always returns a small number would silently bless every broken
gradient in the repository.

These tests therefore work in both directions: correct gradients must pass, and
deliberately corrupted ones must fail.

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

gc = load("gradient_checking.py")
gradient_check = gc.gradient_check
dictionary_to_vector = gc.dictionary_to_vector
vector_to_dictionary = gc.vector_to_dictionary


def quadratic(X, Y, parameters):
    """
    J = scale * sum(theta^2), so dJ/dtheta = 2 * scale * theta.

    The cost depends on X and Y through `scale`, which is what makes this a test
    of the documented `cost_function(X, Y, parameters)` contract and not just of
    the differencing arithmetic.
    """
    scale = float(np.sum(X * Y))
    return scale * sum(float(np.sum(p**2)) for p in parameters.values())


def quadratic_gradients(X, Y, parameters):
    scale = float(np.sum(X * Y))
    return {key: 2 * scale * value for key, value in parameters.items()}


@pytest.fixture
def problem():
    rng = np.random.default_rng(0)
    parameters = {
        "W1": rng.normal(size=(3, 2)),
        "b1": rng.normal(size=(3, 1)),
        "W2": rng.normal(size=(1, 3)),
        "b2": rng.normal(size=(1, 1)),
    }
    X = rng.normal(size=(2, 5))
    Y = rng.normal(size=(2, 5))
    return parameters, X, Y


def test_correct_gradients_score_below_the_excellent_threshold(problem):
    """The docstring promises < 1e-7 means 'likely correct'. Exact gradients must reach it."""
    parameters, X, Y = problem
    difference = gradient_check(
        parameters, quadratic_gradients(X, Y, parameters), X, Y, quadratic
    )
    assert difference < 1e-7, f"exact gradients scored {difference:.2e}"


def test_a_sign_flipped_gradient_is_rejected(problem):
    """
    The defect the checker exists to catch. A flipped sign trains the model
    uphill while every shape and finiteness assertion still passes, so this is
    the one failure mode that must never score as acceptable.
    """
    parameters, X, Y = problem
    wrong = {k: -v for k, v in quadratic_gradients(X, Y, parameters).items()}
    assert gradient_check(parameters, wrong, X, Y, quadratic) > 0.9


def test_a_gradient_that_is_wrong_in_one_entry_is_rejected(problem):
    """
    A single bad entry is the realistic case: one layer's term dropped, the rest
    correct. The relative difference is a norm over all parameters, so it must
    still rise above the 1e-3 'poor' threshold rather than being averaged away.
    """
    parameters, X, Y = problem
    wrong = quadratic_gradients(X, Y, parameters)
    wrong["W1"] = wrong["W1"].copy()
    wrong["W1"][0, 0] += 1.0
    assert gradient_check(parameters, wrong, X, Y, quadratic) > 1e-3


def test_a_uniformly_scaled_gradient_is_rejected(problem):
    """A missing 1/m, the most common backprop slip after a sign error."""
    parameters, X, Y = problem
    wrong = {k: v * 5.0 for k, v in quadratic_gradients(X, Y, parameters).items()}
    assert gradient_check(parameters, wrong, X, Y, quadratic) > 1e-3


def test_the_cost_function_receives_x_y_and_a_parameter_dictionary(problem):
    """
    Pins the documented callback contract. The signature accepts X and Y, so they
    have to reach the callback; a checker that ignored them would force every
    caller to close over its data and would make the two parameters a lie.
    """
    parameters, X, Y = problem
    seen = []

    def spy(X_arg, Y_arg, params_arg):
        seen.append((X_arg, Y_arg, params_arg))
        return quadratic(X_arg, Y_arg, params_arg)

    gradient_check(parameters, quadratic_gradients(X, Y, parameters), X, Y, spy)

    assert seen, "cost_function was never called"
    for X_arg, Y_arg, params_arg in seen:
        assert X_arg is X and Y_arg is Y
        assert set(params_arg) == set(parameters)
        for key, value in params_arg.items():
            assert value.shape == parameters[key].shape


def test_zero_gradients_and_a_flat_cost_return_zero_rather_than_dividing_by_zero(
    problem,
):
    """Both norms vanish, so the relative difference is 0/0. It must not raise."""
    parameters, X, Y = problem
    flat = {key: np.zeros_like(value) for key, value in parameters.items()}
    assert gradient_check(flat, flat, X, Y, lambda X, Y, p: 0.0) == 0.0


def test_the_parameters_are_left_unmodified(problem):
    """
    Differencing perturbs each entry in turn. It must work on copies: a checker
    that leaves a parameter shifted by epsilon corrupts the model it was called
    to verify.
    """
    parameters, X, Y = problem
    before = {key: value.copy() for key, value in parameters.items()}
    gradient_check(parameters, quadratic_gradients(X, Y, parameters), X, Y, quadratic)
    for key, value in before.items():
        np.testing.assert_array_equal(parameters[key], value)


class TestVectorRoundTrip:
    """
    Flattening and unflattening is where a gradient checker silently goes wrong:
    if the two functions disagree on ordering, every parameter is differenced
    against the wrong gradient entry and the result is noise that still looks
    like a number.
    """

    def test_a_dictionary_survives_a_round_trip(self, problem):
        parameters, _, _ = problem
        theta, shapes = dictionary_to_vector(parameters)
        restored = vector_to_dictionary(theta, shapes)

        assert set(restored) == set(parameters)
        for key, value in parameters.items():
            np.testing.assert_array_equal(restored[key], value)

    def test_the_vector_is_a_column_holding_every_element(self, problem):
        parameters, _, _ = problem
        theta, _ = dictionary_to_vector(parameters)
        assert theta.shape == (sum(v.size for v in parameters.values()), 1)

    def test_ordering_is_by_sorted_key_so_two_dictionaries_align(self):
        """
        Parameters and gradients are flattened by separate calls, then subtracted
        elementwise. They only line up if both use the same order, which is why
        the order is sorted keys rather than insertion order.
        """
        first = {"W1": np.array([[1.0]]), "b1": np.array([[2.0]])}
        second = {"b1": np.array([[2.0]]), "W1": np.array([[1.0]])}
        np.testing.assert_array_equal(
            dictionary_to_vector(first)[0], dictionary_to_vector(second)[0]
        )

    def test_shapes_are_reported_not_inferred(self, problem):
        parameters, _, _ = problem
        _, shapes = dictionary_to_vector(parameters)
        assert shapes == {key: value.shape for key, value in parameters.items()}
