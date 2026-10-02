"""
Gradient Checking Utility
=========================

Provides implementation of gradient checking for neural networks:
numerically approximates gradients via finite differences and compares
them to analytical gradients from backpropagation.

This helps validate correctness of gradient computations and debug implementation errors.

This is the *whole-network* checker: hand it a parameter dictionary, the matching
analytic gradients, and a ``cost_function(X, Y, parameters)``; it flattens the
dictionary into one vector, perturbs every entry, and prints a human-readable
verdict. It is the taught, public gradient check, with its own reference,
how-to, and explanation pages.

It is not the only gradient check in the hub, and the two are not
interchangeable. The sequence test suites use a *per-tensor, closure-driven*
harness, :mod:`dlhub.nn.sequence._gradient_check` -- the caller perturbs one
tensor against a zero-argument loss that reads it in place, which is what a
recurrent forward pass over a fixture needs. That harness is deliberately
private (see its module docstring); reach for this module when you want a
network-level verdict, and for that one when you are differentiating a single
tensor inside a test.

References
----------
- Karpathy, A. (n.d.). *Numerical Limits and Gradient Checking*, in "CS231n". Stanford University.
- Ng, A. (2017). *Deep Learning Specialization*: Week 3 – Gradient Checking.

Author
------
Deep Learning Reference Hub

License
-------
MIT License

Notes
-----
- Compute numerical gradient using ε-shift method: (J(θ+ε) - J(θ-ε)) / (2ε).
- Compare with backward-mode gradients using relative difference metric.
- Use small ε (e.g. 1e-7), and expect relative difference < 1e-7.
"""

from collections.abc import Callable

import numpy as np


def gradient_check(
    parameters: dict[str, np.ndarray],
    gradients: dict[str, np.ndarray],
    X: np.ndarray,
    Y: np.ndarray,
    cost_function: Callable[[np.ndarray, np.ndarray, dict[str, np.ndarray]], float],
    epsilon: float = 1e-7,
) -> float:
    """
    Perform gradient checking to verify analytical gradients against numerical gradients.

    Parameters
    ----------
    parameters : dict
        Dictionary of parameters (e.g., {'W1': array, 'b1': array, ...})
    gradients : dict
        Dictionary of computed analytical gradients
    X : np.ndarray
        Input data
    Y : np.ndarray
        True labels
    cost_function : callable
        Function that computes cost given (X, Y, parameters)
    epsilon : float, default=1e-7
        Small value for numerical differentiation

    Returns
    -------
    float
        Relative difference between numerical and analytical gradients
           - < 1e-7: Excellent (gradients are likely correct)
           - < 1e-5: Good (gradients are probably correct)
           - < 1e-3: Acceptable (check implementation)
           - > 1e-3: Poor (likely bug in gradient computation)
    """
    params_vector, param_shapes = dictionary_to_vector(parameters)
    grad_vector, _ = dictionary_to_vector(gradients)

    num_parameters = params_vector.shape[0]
    gradapprox = np.zeros((num_parameters, 1))

    # Each parameter is shifted in a fresh copy of the vector, so the caller's
    # `parameters` is never left holding a perturbed value.
    for i in range(num_parameters):
        theta_plus = np.copy(params_vector)
        theta_plus[i] = theta_plus[i] + epsilon
        J_plus = cost_function(X, Y, vector_to_dictionary(theta_plus, param_shapes))

        theta_minus = np.copy(params_vector)
        theta_minus[i] = theta_minus[i] - epsilon
        J_minus = cost_function(X, Y, vector_to_dictionary(theta_minus, param_shapes))

        gradapprox[i] = (J_plus - J_minus) / (2 * epsilon)  # Numerical Gradient

    # Relative Difference Computation
    numerator = np.linalg.norm(grad_vector - gradapprox)
    denominator = np.linalg.norm(grad_vector) + np.linalg.norm(gradapprox)

    if denominator == 0:
        return 0.0
    difference = numerator / denominator

    print("Gradient Check Results:")
    print(f"  Numerical gradient norm: {np.linalg.norm(gradapprox):.6f}")
    print(f"  Analytical gradient norm: {np.linalg.norm(grad_vector):.6f}")
    print(f"  Relative difference: {difference:.2e}")

    if difference < 1e-7:
        print("  ✅ Excellent! Gradients are likely correct.")
    elif difference < 1e-5:
        print("  ✅ Good! Gradients are probably correct.")
    elif difference < 1e-3:
        print("  ⚠️  Acceptable, but check your implementation.")
    else:
        print("  ❌ Poor! Likely bug in gradient computation.")

    return difference


def dictionary_to_vector(
    parameters: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, tuple]]:
    """
    Convert parameter dictionary to a single vector while preserving shape information.

    Parameters
    ----------
    parameters : dict[str, np.ndarray]
        Dictionary with parameter names as keys and numpy arrays as values

    Returns
    -------
    tuple[np.ndarray, dict[str, tuple]]: (theta, shapes) where:
        - theta: Single column vector containing all parameters
        - shapes: Dictionary mapping parameter names to their original shapes
    """
    shapes = {}
    theta = None

    for key in sorted(parameters.keys()):  # Sort for consistent ordering
        shapes[key] = parameters[key].shape

        param_vector = np.reshape(parameters[key], (-1, 1))
        if theta is None:
            theta = param_vector
        else:
            theta = np.concatenate((theta, param_vector), axis=0)

    return theta, shapes


def vector_to_dictionary(
    theta: np.ndarray, shapes: dict[str, tuple]
) -> dict[str, np.ndarray]:
    """
    Convert a parameter vector back to dictionary format using stored shapes.

    Parameters
    ----------
    theta : np.ndarray
        Column vector containing all parameters
    shapes : dict[str, tuple]
        Dictionary mapping parameter names to their original shapes

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with parameter names as keys and reshaped arrays as values
    """
    parameters = {}
    start = 0

    for key in sorted(shapes.keys()):
        shape = shapes[key]
        size = np.prod(shape)  # Total number of elements

        parameters[key] = theta[start : start + size].reshape(shape)
        start += size

    return parameters
