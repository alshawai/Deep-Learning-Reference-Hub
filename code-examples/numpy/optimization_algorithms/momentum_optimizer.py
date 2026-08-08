"""
Momentum Optimizer Implementation
=================================

This module implements gradient descent with momentum, including exponential
weighted averages and bias correction for efficient neural network training.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import numpy as np
from typing import Dict, Any, Tuple


class MomentumOptimizer:
    """
    Gradient Descent with Momentum optimizer.

    This implementation uses exponential weighted averages to accumulate gradients
    and includes bias correction for better convergence, especially in early training.

    Parameters
    ----------
    learning_rate : float, default=0.001
        Learning rate for parameter updates
    beta : float, default=0.9
        Momentum parameter (exponential decay rate)
    bias_correction : bool, default=True
        Whether to apply bias correction to momentum estimates
    epsilon : float, default=1e-8
        Small constant for numerical stability

    Attributes
    ----------
    learning_rate : float
        Current learning rate
    beta : float
        Momentum parameter
    bias_correction : bool
        Bias correction flag
    epsilon : float
        Numerical stability constant
    v : Dict[str, np.ndarray]
        Momentum (velocity) estimates for each parameter
    t : int
        Time step counter for bias correction
    history : Dict[str, List[float]]
        Training history
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta: float = 0.9,
        bias_correction: bool = True,
        epsilon: float = 1e-8,
    ):
        self.learning_rate = learning_rate
        self.beta = beta
        self.bias_correction = bias_correction
        self.epsilon = epsilon

        self.v = {}  # Momentum estimates
        self.t = 0  # Time step
        self.history = {"loss": [], "gradient_norm": []}

    def initialize_velocity(self, parameters: Dict[str, np.ndarray]) -> None:
        """
        Initialize velocity (momentum) estimates for all parameters.

        Parameters
        ----------
        parameters : Dict[str, np.ndarray]
            Model parameters to initialize velocity for

        Notes
        -----
        Velocities are initialized to zero arrays with the same shape as parameters.
        """
        for key in parameters:
            self.v[key] = np.zeros_like(parameters[key])

    def update_parameters(
        self, parameters: Dict[str, np.ndarray], gradients: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Update parameters using momentum-based gradient descent.

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
        Updates parameters using momentum:
        v_t = β * v_{t-1} + (1-β) * g_t
        θ_t = θ_{t-1} - α * v_t_corrected

        Where v_t_corrected includes bias correction if enabled.
        """
        if not self.v:
            self.initialize_velocity(parameters)

        self.t += 1
        updated_parameters = {}

        for key in parameters:
            self.v[key] = self.beta * self.v[key] + (1 - self.beta) * gradients[key]

            if self.bias_correction:
                v_corrected = self.v[key] / (1 - self.beta**self.t)
            else:
                v_corrected = self.v[key]

            updated_parameters[key] = parameters[key] - self.learning_rate * v_corrected

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

    def train_step(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        parameters: Dict[str, np.ndarray],
        forward_propagation_fn: callable,
        backward_propagation_fn: callable,
        compute_cost_fn: callable,
    ) -> Tuple[Dict[str, np.ndarray]]:
        """
        Perform one training step with momentum optimizer.

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
        Train the model using momentum optimizer.

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
                print(
                    f"Epoch {epoch}: Cost = {cost:.6f}, Gradient Norm = {grad_norm:.6f}"
                )

        return parameters

    def get_momentum_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics about momentum estimates.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Statistics for each parameter's momentum
        """
        stats = {}
        for key, v in self.v.items():
            stats[key] = {
                "mean": np.mean(v),
                "std": np.std(v),
                "max": np.max(v),
                "min": np.min(v),
                "norm": np.linalg.norm(v),
            }
        return stats

    def reset_optimizer_state(self) -> None:
        """Reset optimizer state including velocity estimates and time step."""
        self.v = {}
        self.t = 0
        self.history = {"loss": [], "gradient_norm": []}

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
            "beta": self.beta,
            "bias_correction": self.bias_correction,
            "epsilon": self.epsilon,
            "optimizer": "MomentumOptimizer",
        }


def example_usage():
    """Example demonstrating how to use MomentumOptimizer."""
    np.random.seed(42)
    X = np.random.randn(10, 1000)  # 10 features, 1000 examples
    Y = (X[0:1, :] > 0).astype(int)  # Binary classification

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
        return {
            key: np.random.randn(*val.shape) * 0.01 for key, val in parameters.items()
        }

    def compute_cost(AL, Y):
        return np.random.rand()

    optimizer = MomentumOptimizer(learning_rate=0.01, beta=0.9, bias_correction=True)

    trained_parameters = optimizer.fit(
        X,
        Y,
        parameters,
        forward_propagation,
        backward_propagation,
        compute_cost,
        epochs=100,
        print_every=20,
    )

    print(f"\nFinal cost: {optimizer.history['loss'][-1]:.6f}")
    print(f"Momentum statistics: \n{optimizer.get_momentum_statistics()}")
    print(f"\nOptimizer config: {optimizer.get_config()}")


if __name__ == "__main__":
    example_usage()
