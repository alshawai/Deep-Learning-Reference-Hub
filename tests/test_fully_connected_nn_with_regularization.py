"""
Tests for the L-layer fully connected network.

The centrepiece is a finite-difference gradient check. Backpropagation is the
one place in this repository where a sign error or a missing term produces
output of the right shape, in the right range, that trains to a plausible-looking
loss curve and is still wrong. Shape assertions cannot see it. Comparing the
analytic gradient against a numerical one can.
"""

import numpy as np
import pytest

from dlhub.nn.fully_connected import DeepNeuralNetwork


def tiny_problem(n_features=4, m=12, seed=0):
    """A separable binary problem small enough to difference every parameter."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_features, m))
    Y = (X[0] + X[1] > 0).astype(float).reshape(1, m)
    return X, Y


def numerical_gradient(net, X, Y, key, index, eps=1e-7):
    """Two-sided difference of the cost with respect to one scalar parameter."""
    original = net.parameters[key][index]

    net.parameters[key][index] = original + eps
    AL, _ = net.forward_propagation(X, training=False)
    cost_plus = net.compute_cost(AL, Y)

    net.parameters[key][index] = original - eps
    AL, _ = net.forward_propagation(X, training=False)
    cost_minus = net.compute_cost(AL, Y)

    net.parameters[key][index] = original
    return (cost_plus - cost_minus) / (2 * eps)


@pytest.mark.parametrize("layer_dims", [[4, 1], [4, 5, 1], [4, 5, 3, 1]])
def test_backprop_matches_finite_differences(layer_dims):
    """
    Analytic and numerical gradients must agree to ~1e-7 relative.

    Regularization and dropout are off: L2 adds a term the cost function must
    also see for the comparison to be valid, and dropout makes the forward pass
    stochastic, which breaks differencing outright.
    """
    X, Y = tiny_problem(n_features=layer_dims[0])
    net = DeepNeuralNetwork(
        layer_dims, regularization=None, keep_prob=1.0, use_batch_norm=False
    )

    AL, caches = net.forward_propagation(X, training=False)
    grads = net.backward_propagation(AL, Y, caches)

    analytic, numeric = [], []
    for l in range(1, len(layer_dims)):
        for key in (f"W{l}", f"b{l}"):
            flat = net.parameters[key]
            for index in np.ndindex(flat.shape):
                analytic.append(grads[f"d{key}"][index])
                numeric.append(numerical_gradient(net, X, Y, key, index))

    analytic = np.array(analytic)
    numeric = np.array(numeric)
    denom = np.linalg.norm(analytic) + np.linalg.norm(numeric)
    relative_error = np.linalg.norm(analytic - numeric) / denom
    assert relative_error < 1e-7, f"gradient check failed: {relative_error:.2e}"


def test_l2_gradient_matches_finite_differences_of_the_regularized_cost():
    """
    With L2 on, the weight gradient carries an extra lambda_reg/m * W term. The
    check only closes if compute_cost adds the matching penalty, so this pins
    the two halves of L2 together.
    """
    X, Y = tiny_problem()
    net = DeepNeuralNetwork(
        [4, 5, 1],
        regularization="l2",
        lambda_reg=0.7,
        keep_prob=1.0,
        use_batch_norm=False,
    )

    AL, caches = net.forward_propagation(X, training=False)
    grads = net.backward_propagation(AL, Y, caches)

    errors = []
    for l in (1, 2):
        key = f"W{l}"
        for index in np.ndindex(net.parameters[key].shape):
            errors.append(
                abs(grads[f"d{key}"][index] - numerical_gradient(net, X, Y, key, index))
            )
    assert max(errors) < 1e-6, f"L2 gradient mismatch: {max(errors):.2e}"


def test_gradients_stay_finite_at_every_depth_with_batch_norm():
    """
    Regression test for a loop-carried `dA_next`, which was bound only inside a
    guarded branch of the previous iteration. It happened to work, and it was one
    edit from a NameError that no test covered.
    """
    for depth in range(1, 5):
        layer_dims = [4] + [5] * (depth - 1) + [1]
        for batch_norm in (False, True):
            X, Y = tiny_problem()
            net = DeepNeuralNetwork(
                layer_dims, use_batch_norm=batch_norm, keep_prob=1.0
            )
            AL, caches = net.forward_propagation(X, training=True)
            grads = net.backward_propagation(AL, Y, caches)
            for name, value in grads.items():
                assert np.all(np.isfinite(value)), f"{name} not finite at depth {depth}"


def test_he_initialization_variance_follows_2_over_fan_in():
    """He scaling is 2/fan_in; getting it wrong is invisible until depth kills the signal."""
    net = DeepNeuralNetwork([500, 400, 1], initialization="he")
    observed = net.parameters["W1"].std()
    np.testing.assert_allclose(observed, np.sqrt(2.0 / 500), rtol=0.05)


def test_xavier_initialization_variance_follows_1_over_fan_in():
    net = DeepNeuralNetwork([500, 400, 1], initialization="xavier")
    observed = net.parameters["W2"].std()
    np.testing.assert_allclose(observed, np.sqrt(1.0 / 400), rtol=0.05)


def test_biases_start_at_zero():
    net = DeepNeuralNetwork([4, 5, 1])
    for l in (1, 2):
        assert np.all(net.parameters[f"b{l}"] == 0.0)


def test_parameter_shapes_follow_the_layer_dims():
    layer_dims = [7, 5, 3, 1]
    net = DeepNeuralNetwork(layer_dims)
    for l in range(1, len(layer_dims)):
        assert net.parameters[f"W{l}"].shape == (layer_dims[l], layer_dims[l - 1])
        assert net.parameters[f"b{l}"].shape == (layer_dims[l], 1)


def test_l2_penalty_raises_the_cost_and_no_regularization_does_not():
    """The penalty is lambda_reg/(2m) * sum(W^2), so it is strictly positive for lambda_reg > 0."""
    X, Y = tiny_problem()
    plain = DeepNeuralNetwork([4, 5, 1], regularization=None, keep_prob=1.0)
    AL, _ = plain.forward_propagation(X, training=False)
    unpenalized = plain.compute_cost(AL, Y)

    regularized = DeepNeuralNetwork(
        [4, 5, 1], regularization="l2", lambda_reg=1.0, keep_prob=1.0
    )
    regularized.parameters = plain.parameters
    AL, _ = regularized.forward_propagation(X, training=False)
    penalized = regularized.compute_cost(AL, Y)

    assert penalized > unpenalized


def test_training_reduces_the_cost_on_a_separable_problem():
    """
    An end-to-end sanity check on the sign of the update. A backwards step raises
    the cost, which no shape or finiteness assertion detects.
    """
    X, Y = tiny_problem(m=60)
    net = DeepNeuralNetwork([4, 8, 1], keep_prob=1.0)
    history = net.train(
        X,
        Y,
        X,
        Y,
        learning_rate=0.1,
        num_epochs=300,
        print_cost=False,
        early_stopping=False,
    )
    costs = history["costs"]
    assert costs[-1] < costs[0], "cost did not decrease"


def test_predictions_are_binary_and_correctly_shaped():
    X, Y = tiny_problem(m=20)
    net = DeepNeuralNetwork([4, 5, 1], keep_prob=1.0)
    predictions = net.predict(X)
    assert predictions.shape == (1, 20)
    assert set(np.unique(predictions)).issubset({0, 1})


def test_dropout_is_stochastic_in_training_and_absent_at_inference():
    """
    Inference must be deterministic. A dropout mask left on at test time makes
    predictions vary between calls on identical input, which is a silent defect:
    the accuracy only drops a little, and nothing raises.

    The mask is only wired up when `regularization="dropout"`, so this also pins
    that `keep_prob` alone does not quietly enable it.
    """
    X, _ = tiny_problem()
    net = DeepNeuralNetwork([4, 20, 1], regularization="dropout", keep_prob=0.5)

    first, _ = net.forward_propagation(X, training=False)
    second, _ = net.forward_propagation(X, training=False)
    np.testing.assert_array_equal(first, second)

    trained_passes = [net.forward_propagation(X, training=True)[0] for _ in range(2)]
    assert not np.array_equal(*trained_passes), "dropout mask is not being sampled"


def test_keep_prob_alone_does_not_enable_dropout():
    """Without `regularization="dropout"` the forward pass is deterministic in training too."""
    X, _ = tiny_problem()
    net = DeepNeuralNetwork([4, 20, 1], regularization=None, keep_prob=0.5)
    first, _ = net.forward_propagation(X, training=True)
    second, _ = net.forward_propagation(X, training=True)
    np.testing.assert_array_equal(first, second)
