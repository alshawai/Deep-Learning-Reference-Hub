"""
RMSprop Optimizer Implementation
=================================

This module implements the RMSprop (Root Mean Square Propagation) optimizer
with adaptive learning rate optimization using squared gradient accumulation
and parameter-wise learning rate adjustment.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import numpy as np
from typing import Dict, Any, Tuple


class RMSpropOptimizer:
    """
    RMSprop (Root Mean Square Propagation) optimizer.

    RMSprop adapts the learning rate for each parameter by dividing by a running
    average of the magnitudes of recent gradients. This helps with convergence
    on non-convex functions and handles different scaling of parameters. It also
    applies a learning rate decay technique based on current step.

    Parameters
    ----------
    learning_rate : float, default=0.001
        Learning rate for parameter updates
    beta : float, default=0.9
        Exponential decay rate for the second moment estimates
    epsilon : float, default=1e-8
        Small constant for numerical stability
    bias_correction : bool, default=False
        Whether to apply bias correction (not standard in RMSprop)
    decay : float, default=0.0
        Learning rate decay factor

    Attributes
    ----------
    learning_rate : float
        Current learning rate
    beta : float
        Decay rate for second moment estimates
    epsilon : float
        Numerical stability constant
    s : Dict[str, np.ndarray]
        Second moment estimates (squared gradients) for each parameter
    t : int
        Time step counter
    history : Dict[str, List[float]]
        Training history including losses and learning rates
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta: float = 0.9,
        epsilon: float = 1e-8,
        bias_correction: bool = False,
        decay: float = 0.0,
    ):
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.bias_correction = bias_correction
        self.decay = decay

        self.s = {}  # Second moment estimates
        self.t = 0  # Time step
        self.history = {
            "loss": [],
            "gradient_norm": [],
            "learning_rate": [],
            "rms_grad": [],
        }

    def initialize_second_moments(self, parameters: Dict[str, np.ndarray]) -> None:
        """
        Initialize second moment estimates for all parameters.

        Parameters
        ----------
        parameters : Dict[str, np.ndarray]
            Model parameters to initialize second moments for

        Notes
        -----
        Second moments are initialized to zero arrays with the same shape as parameters.
        """
        for key in parameters:
            self.s[key] = np.zeros_like(parameters[key])

    def update_learning_rate(self) -> None:
        """
        Update learning rate with decay if specified.

        Notes
        -----
        Applies learning rate decay: lr = lr_initial / (1 + decay * t)
        """
        if self.decay > 0:
            self.learning_rate = self.initial_learning_rate / (1 + self.decay * self.t)

    def update_parameters(
        self, parameters: Dict[str, np.ndarray], gradients: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Update parameters using RMSprop optimization.

        Parameters
        ----------
        parameters : Dict[str, np.ndarray]
            Current model parameters
        gradients : Dict[str, np.ndarray]
            Computed gradients for each parameter

        Returns
        -------
        Dict[str, np.ndarray]
            Updated parameters

        Notes
        -----
        Updates parameters using RMSprop:
        s_t = β * s_{t-1} + (1-β) * g_t²
        θ_t = θ_{t-1} - α * g_t / (√s_t + ε)

        Where s_t is the exponential weighted average of squared gradients.
        Stores average RMS gradient for monitoring
        """
        if not self.s:
            self.initialize_second_moments(parameters)

        self.t += 1
        self.update_learning_rate()

        updated_parameters = {}
        rms_gradients = {}

        for key in parameters:
            self.s[key] = self.beta * self.s[key] + (1 - self.beta) * (
                gradients[key] ** 2
            )

            if self.bias_correction:
                s_corrected = self.s[key] / (1 - self.beta**self.t)
            else:
                s_corrected = self.s[key]

            rms_grad = np.sqrt(np.mean(s_corrected))
            rms_gradients[key] = rms_grad

            updated_parameters[key] = parameters[key] - self.learning_rate * gradients[
                key
            ] / (np.sqrt(s_corrected) + self.epsilon)

        avg_rms_grad = np.mean(list(rms_gradients.values()))
        self.history["rms_grad"].append(avg_rms_grad)
        self.history["learning_rate"].append(self.learning_rate)

        return updated_parameters

    def compute_gradient_norm(self, gradients: Dict[str, np.ndarray]) -> float:
        """
        Compute the L2 norm of gradients for monitoring convergence.

        Parameters
        ----------
        gradients : Dict[str, np.ndarray]
            Gradients for each parameter

        Returns
        -------
        float
            L2 norm of all gradients
        """
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += np.sum(grad**2)
        return np.sqrt(total_norm)

    def get_effective_learning_rates(
        self, parameters: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Compute effective learning rates for each parameter.

        Parameters
        ----------
        parameters : Dict[str, np.ndarray]
            Current model parameters

        Returns
        -------
        Dict[str, np.ndarray]
            Effective learning rates for each parameter
        """
        effective_lrs = {}

        if not self.s:
            for key in parameters:  # If not initialized, return base learning rate
                effective_lrs[key] = np.full_like(parameters[key], self.learning_rate)
        else:
            for key in parameters:
                if self.bias_correction:
                    s_corrected = self.s[key] / (1 - self.beta**self.t)
                else:
                    s_corrected = self.s[key]

                effective_lrs[key] = self.learning_rate / (
                    np.sqrt(s_corrected) + self.epsilon
                )

        return effective_lrs

    def train_step(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        parameters: Dict[str, np.ndarray],
        forward_propagation_fn: callable,
        backward_propagation_fn: callable,
        compute_cost_fn: callable,
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Perform one training step with RMSprop optimizer.

        Parameters
        ----------
        X : np.ndarray
            Input features
        Y : np.ndarray
            Target labels
        parameters : Dict[str, np.ndarray]
            Current model parameters
        forward_propagation_fn : callable
            Function to compute forward propagation
        backward_propagation_fn : callable
            Function to compute backward propagation
        compute_cost_fn : callable
            Function to compute cost/loss

        Returns
        -------
        Tuple[Dict[str, np.ndarray], float]
            Updated parameters and current loss
        """
        AL, caches = forward_propagation_fn(X, parameters)
        cost = compute_cost_fn(AL, Y)
        gradients = backward_propagation_fn(AL, Y, caches)
        parameters = self.update_parameters(parameters, gradients)

        grad_norm = self.compute_gradient_norm(gradients)
        self.history["gradient_norm"].append(grad_norm)

        return parameters, cost

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        parameters: Dict[str, np.ndarray],
        forward_propagation_fn: callable,
        backward_propagation_fn: callable,
        compute_cost_fn: callable,
        epochs: int = 1000,
        print_cost: bool = True,
        print_every: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Train the model using RMSprop optimizer.

        Parameters
        ----------
        X : np.ndarray
            Input features
        Y : np.ndarray
            Target labels
        parameters : Dict[str, np.ndarray]
            Initial model parameters
        forward_propagation_fn : callable
            Function to compute forward propagation
        backward_propagation_fn : callable
            Function to compute backward propagation
        compute_cost_fn : callable
            Function to compute cost/loss
        epochs : int, default=1000
            Number of training epochs
        print_cost : bool, default=True
            Whether to print cost during training
        print_every : int, default=100
            Print cost every N epochs

        Returns
        -------
        Dict[str, np.ndarray]
            Trained parameters
        """
        for epoch in range(epochs):
            parameters, cost = self.train_step(
                X,
                Y,
                parameters,
                forward_propagation_fn,
                backward_propagation_fn,
                compute_cost_fn,
            )

            self.history["loss"].append(cost)

            if print_cost and epoch % print_every == 0:
                grad_norm = self.history["gradient_norm"][-1]
                rms_grad = self.history["rms_grad"][-1]
                lr = self.history["learning_rate"][-1]
                print(
                    f"Epoch {epoch}: Cost = {cost:.6f}, "
                    f"Gradient Norm = {grad_norm:.6f}, "
                    f"RMS Grad = {rms_grad:.6f}, "
                    f"LR = {lr:.6f}"
                )

        return parameters

    def get_second_moment_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics about second moment estimates.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Statistics for each parameter's second moments
        """
        stats = {}
        for key, s in self.s.items():
            stats[key] = {
                "mean": np.mean(s),
                "std": np.std(s),
                "max": np.max(s),
                "min": np.min(s),
                "norm": np.linalg.norm(s),
            }
        return stats

    def reset_optimizer_state(self) -> None:
        """Reset optimizer state including second moment estimates and time step."""
        self.s = {}
        self.t = 0
        self.learning_rate = self.initial_learning_rate
        self.history = {
            "loss": [],
            "gradient_norm": [],
            "learning_rate": [],
            "rms_grad": [],
        }

    def get_config(self) -> Dict[str, Any]:
        """
        Get optimizer configuration.

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary
        """
        return {
            "learning_rate": self.learning_rate,
            "initial_learning_rate": self.initial_learning_rate,
            "beta": self.beta,
            "epsilon": self.epsilon,
            "bias_correction": self.bias_correction,
            "decay": self.decay,
            "optimizer": "RMSpropOptimizer",
        }


def adaptive_learning_rate_analysis(
    optimizer: RMSpropOptimizer, parameters: Dict[str, np.ndarray]
) -> None:
    """
    Analyze and visualize adaptive learning rates in RMSprop.

    Parameters
    ----------
    optimizer : RMSpropOptimizer
        Trained RMSprop optimizer
    parameters : Dict[str, np.ndarray]
        Model parameters
    """
    print("=== Adaptive Learning Rate Analysis ===")
    effective_lrs = optimizer.get_effective_learning_rates(parameters)

    for key, lr_array in effective_lrs.items():
        print(f"\nParameter {key}:")
        print(f"  Base LR: {optimizer.learning_rate:.6f}")
        print(f"  Effective LR - Mean: {np.mean(lr_array):.6f}")
        print(f"  Effective LR - Std: {np.std(lr_array):.6f}")
        print(f"  Effective LR - Min: {np.min(lr_array):.6f}")
        print(f"  Effective LR - Max: {np.max(lr_array):.6f}")
        print(f"  Adaptation Ratio: {np.mean(lr_array) / optimizer.learning_rate:.4f}")


def example_usage():
    """Example demonstrating how to use RMSpropOptimizer."""
    np.random.seed(42)
    X = np.random.randn(10, 1000)  # 10 features, 1000 examples
    Y = (X[0:1, :] > 0).astype(int)  # Binary classification

    X[5:7, :] *= 10  # Larger scale features!
    X[8:, :] *= 0.1  # Smaller scale features!

    def initialize_parameters(layer_dims):
        parameters = {}
        L = len(layer_dims)
        for l in range(1, L):
            parameters[f"W{l}"] = (
                np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
            )
            parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))
        return parameters

    layer_dims = [10, 5, 1]
    parameters = initialize_parameters(layer_dims)

    # Dummy functions (replace with actual implementations)
    def forward_propagation(X, parameters):
        return np.random.randn(1, X.shape[1]), {}

    def backward_propagation(AL, Y, caches):
        gradients = {}  # Simulating gradients with different scales
        for key, val in parameters.items():
            if "W" in key:
                gradients[key] = np.random.randn(*val.shape) * 0.1
            else:  # bias terms
                gradients[key] = np.random.randn(*val.shape) * 0.01
        return gradients

    def compute_cost(AL, Y):
        return np.random.rand()

    optimizer = RMSpropOptimizer(
        learning_rate=0.001, beta=0.9, epsilon=1e-8, bias_correction=False, decay=1e-3
    )

    trained_parameters = optimizer.fit(
        X,
        Y,
        parameters,
        forward_propagation,
        backward_propagation,
        compute_cost,
        epochs=1000,
        print_every=100,
    )

    print(f"\nFinal cost: {optimizer.history['loss'][-1]:.6f}")
    print(f"Final learning rate: {optimizer.learning_rate:.6f}")
    print(f"\nSecond moment statistics: {optimizer.get_second_moment_statistics()}")
    print(f"\nOptimizer config:\n{optimizer.get_config()}\n")

    adaptive_learning_rate_analysis(optimizer, trained_parameters)


if __name__ == "__main__":
    example_usage()
