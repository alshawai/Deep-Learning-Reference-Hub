"""
Mini-batch Gradient Descent Tests
=================================

Batching looks trivial and has one failure mode that matters: X and Y must be
permuted together. Shuffling them independently trains the model on scrambled
labels, which does not raise, does not change any shape, and simply fails to
learn. Several tests here exist only to pin that alignment.

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

mbgd = load("optimization_algorithms/mini_batch_gradient_descent.py")
MiniBatchGradientDescent = mbgd.MiniBatchGradientDescent
initialize_parameters = mbgd.initialize_parameters


@pytest.fixture
def labelled_data():
    """
    Y carries each column's original index, so alignment is checkable after any
    permutation: column j of X must always arrive with the label j.
    """
    m = 23
    X = np.arange(3 * m, dtype=float).reshape(3, m)
    Y = np.arange(m, dtype=float).reshape(1, m)
    return X, Y


class TestBatching:
    def test_batches_partition_the_data(self, labelled_data):
        X, Y = labelled_data
        opt = MiniBatchGradientDescent(batch_size=5, random_seed=0)
        batches = opt.create_mini_batches(X, Y)

        assert sum(xb.shape[1] for xb, _ in batches) == X.shape[1]
        labels = np.concatenate([yb.ravel() for _, yb in batches])
        np.testing.assert_array_equal(np.sort(labels), Y.ravel())

    def test_features_and_labels_stay_aligned_through_the_shuffle(self, labelled_data):
        """
        The defect this file exists for. Permuting X and Y with two separate
        calls produces batches of the right shapes holding mismatched pairs.
        """
        X, Y = labelled_data
        opt = MiniBatchGradientDescent(batch_size=4, shuffle=True, random_seed=0)
        for X_batch, Y_batch in opt.create_mini_batches(X, Y):
            for column in range(X_batch.shape[1]):
                index = int(Y_batch[0, column])
                np.testing.assert_array_equal(X_batch[:, column], X[:, index])

    def test_the_last_batch_holds_the_remainder(self, labelled_data):
        X, Y = labelled_data  # 23 columns
        opt = MiniBatchGradientDescent(batch_size=5, shuffle=False)
        batches = opt.create_mini_batches(X, Y)
        assert [xb.shape[1] for xb in (b[0] for b in batches)] == [5, 5, 5, 5, 3]

    def test_an_exact_division_produces_no_short_batch(self):
        X, Y = np.zeros((2, 20)), np.zeros((1, 20))
        opt = MiniBatchGradientDescent(batch_size=5, shuffle=False)
        batches = opt.create_mini_batches(X, Y)
        assert len(batches) == 4
        assert all(xb.shape[1] == 5 for xb, _ in batches)

    def test_a_batch_larger_than_the_dataset_yields_one_batch(self):
        X, Y = np.zeros((2, 7)), np.zeros((1, 7))
        opt = MiniBatchGradientDescent(batch_size=100, shuffle=False)
        batches = opt.create_mini_batches(X, Y)
        assert len(batches) == 1
        assert batches[0][0].shape[1] == 7

    def test_shuffle_false_preserves_the_original_order(self, labelled_data):
        X, Y = labelled_data
        opt = MiniBatchGradientDescent(batch_size=5, shuffle=False)
        batches = opt.create_mini_batches(X, Y)
        np.testing.assert_array_equal(
            np.concatenate([yb.ravel() for _, yb in batches]), Y.ravel()
        )

    def test_shuffle_true_actually_reorders(self, labelled_data):
        X, Y = labelled_data
        opt = MiniBatchGradientDescent(batch_size=5, shuffle=True, random_seed=1)
        order = np.concatenate([yb.ravel() for _, yb in opt.create_mini_batches(X, Y)])
        assert not np.array_equal(order, Y.ravel())

    def test_the_inputs_are_not_modified(self, labelled_data):
        X, Y = labelled_data
        before_x, before_y = X.copy(), Y.copy()
        MiniBatchGradientDescent(batch_size=4, random_seed=0).create_mini_batches(X, Y)
        np.testing.assert_array_equal(X, before_x)
        np.testing.assert_array_equal(Y, before_y)


class TestUpdate:
    def test_the_update_is_theta_minus_lr_times_grad(self):
        opt = MiniBatchGradientDescent(learning_rate=0.1)
        updated = opt.update_parameters(
            {"W1": np.array([1.0, 2.0]), "b1": np.array([0.5])},
            {"W1": np.array([10.0, -10.0]), "b1": np.array([1.0])},
        )
        np.testing.assert_allclose(updated["W1"], [0.0, 3.0])
        np.testing.assert_allclose(updated["b1"], [0.4])

    def test_the_original_parameters_are_left_alone(self):
        """The method returns a new dict, so callers can keep the previous step."""
        opt = MiniBatchGradientDescent(learning_rate=0.1)
        parameters = {"W1": np.array([1.0])}
        updated = opt.update_parameters(parameters, {"W1": np.array([1.0])})
        np.testing.assert_array_equal(parameters["W1"], [1.0])
        assert updated["W1"] is not parameters["W1"]


def linear_model():
    """
    A one-parameter least-squares problem: J = mean((wx - y)^2), dJ/dw known in
    closed form. Small enough that the epoch cost can be predicted by hand.
    """

    def forward(X, parameters):
        AL = parameters["w"] @ X
        return AL, (X,)

    def cost(AL, Y):
        return float(np.mean((AL - Y) ** 2))

    def backward(AL, Y, caches):
        (X,) = caches
        m = X.shape[1]
        return {"w": (2.0 / m) * (AL - Y) @ X.T}

    return forward, backward, cost


class TestTraining:
    def test_one_epoch_visits_every_batch(self):
        forward, backward, cost = linear_model()
        seen = []

        def counting_forward(X, parameters):
            seen.append(X.shape[1])
            return forward(X, parameters)

        opt = MiniBatchGradientDescent(learning_rate=0.0, batch_size=4, shuffle=False)
        opt.train_epoch(
            np.ones((1, 10)),
            np.ones((1, 10)),
            {"w": np.zeros((1, 1))},
            counting_forward,
            backward,
            cost,
        )
        assert seen == [4, 4, 2]

    def test_the_epoch_cost_is_the_mean_over_batches(self):
        """
        Not the sum. A cost that grows with the batch count looks like divergence
        the moment anyone changes `batch_size`.
        """
        forward, backward, _ = linear_model()
        opt = MiniBatchGradientDescent(learning_rate=0.0, batch_size=4, shuffle=False)
        _, epoch_cost = opt.train_epoch(
            np.ones((1, 10)),
            np.ones((1, 10)),
            {"w": np.zeros((1, 1))},
            forward,
            backward,
            lambda AL, Y: 3.0,
        )
        assert epoch_cost == pytest.approx(3.0)

    def test_a_zero_learning_rate_leaves_the_parameters_alone(self):
        forward, backward, cost = linear_model()
        opt = MiniBatchGradientDescent(learning_rate=0.0, batch_size=4, shuffle=False)
        parameters, _ = opt.train_epoch(
            np.ones((1, 8)),
            np.ones((1, 8)),
            {"w": np.array([[0.3]])},
            forward,
            backward,
            cost,
        )
        np.testing.assert_allclose(parameters["w"], [[0.3]])

    def test_fit_drives_the_cost_down_on_a_linear_problem(self):
        forward, backward, cost = linear_model()
        rng = np.random.default_rng(0)
        X = rng.normal(size=(1, 64))
        Y = 2.0 * X

        opt = MiniBatchGradientDescent(learning_rate=0.05, batch_size=16, random_seed=0)
        trained = opt.fit(
            X,
            Y,
            {"w": np.zeros((1, 1))},
            forward,
            backward,
            cost,
            epochs=200,
            print_cost=False,
        )
        np.testing.assert_allclose(trained["w"], [[2.0]], atol=1e-3)
        assert opt.history["loss"][-1] < opt.history["loss"][0]

    def test_fit_records_one_loss_per_epoch(self):
        forward, backward, cost = linear_model()
        opt = MiniBatchGradientDescent(learning_rate=0.01, batch_size=8, random_seed=0)
        opt.fit(
            np.ones((1, 16)),
            np.ones((1, 16)),
            {"w": np.zeros((1, 1))},
            forward,
            backward,
            cost,
            epochs=7,
            print_cost=False,
        )
        assert len(opt.history["loss"]) == 7


def test_initialize_parameters_uses_he_scaling_and_zero_biases():
    layer_dims = [400, 300, 1]
    parameters = initialize_parameters(layer_dims)
    for l in range(1, len(layer_dims)):
        assert parameters[f"W{l}"].shape == (layer_dims[l], layer_dims[l - 1])
        assert parameters[f"b{l}"].shape == (layer_dims[l], 1)
        assert np.all(parameters[f"b{l}"] == 0.0)
    np.testing.assert_allclose(parameters["W1"].std(), np.sqrt(2.0 / 400), rtol=0.05)


def test_the_config_reports_the_settings():
    config = MiniBatchGradientDescent(learning_rate=0.5, batch_size=8).get_config()
    assert config["learning_rate"] == 0.5
    assert config["batch_size"] == 8
    assert config["optimizer"] == "MiniBatchGradientDescent"
