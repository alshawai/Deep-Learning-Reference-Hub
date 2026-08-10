"""
Mini-batch Gradient Descent Implementation
==========================================

This module implements efficient mini-batch gradient descent with proper shuffling
and batch creation for neural network training.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import random
from typing import Any

import numpy as np


class MiniBatchGradientDescent:
    """
    Mini-batch Gradient Descent optimizer with configurable batch size and shuffling.

    This implementation provides efficient mini-batch processing with proper data
    shuffling and batch creation for neural network training.

    Parameters
    ----------
    learning_rate : float, default=0.001
        Learning rate for gradient descent updates
    batch_size : int, default=64
        Size of mini-batches for training
    shuffle : bool, default=True
        Whether to shuffle data at each epoch
    random_seed : int, optional
        Random seed for reproducibility

    Attributes
    ----------
    learning_rate : float
        Current learning rate
    batch_size : int
        Mini-batch size
    shuffle : bool
        Shuffling flag
    history : dict
        Training history including losses and metrics
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        shuffle: bool = True,
        random_seed: int | None = None,
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.history = {"loss": [], "accuracy": []}

        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

    def create_mini_batches(
        self, X: np.ndarray, Y: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Create mini-batches from training data with optional shuffling.

        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_features, m_examples)
        Y : np.ndarray
            Target labels of shape (n_classes, m_examples)

        Returns
        -------
        List[Tuple[np.ndarray, np.ndarray]]
            List of (X_batch, Y_batch) tuples

        Notes
        -----
        If shuffle is True, data is randomly permuted before creating batches.
        The last batch may be smaller if the dataset size is not divisible by batch_size.
        """
        m = X.shape[1]
        mini_batchs = []

        if self.shuffle:
            permutation = np.random.permutation(m)
            X_shuffled = X[:, permutation]
            Y_shuffled = Y[:, permutation]
        else:
            X_shuffled = X
            Y_shuffled = Y

        full_batches = m // self.batch_size
        for k in range(full_batches):
            start = k * self.batch_size
            end = start + self.batch_size

            X_batch = X_shuffled[:, start:end]
            Y_batch = Y_shuffled[:, start:end]

            mini_batchs.append((X_batch, Y_batch))

        if m % self.batch_size:
            start = full_batches * self.batch_size
            X_batch = X_shuffled[:, start:]
            Y_batch = Y_shuffled[:, start:]
            mini_batchs.append((X_batch, Y_batch))

        return mini_batchs

    def update_parameters(
        self, parameters: dict[str, np.ndarray], gradients: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """
        Update model parameters using gradient descent.

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
        Updates parameters using the standard gradient descent rule:
        θ = θ - α * ∇J(θ)
        """
        updated_parameters = {}

        for key in parameters:
            updated_parameters[key] = (
                parameters[key] - self.learning_rate * gradients[key]
            )

        return updated_parameters

    def train_epoch(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        parameters: dict[str, np.ndarray],
        forward_propagation_fn: callable,
        backward_propagation_fn: callable,
        compute_cost_fn: callable,
    ) -> tuple[dict[str, np.ndarray], float]:
        """
        Train for one epoch using mini-batch gradient descent.

        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_features, m_examples)
        Y : np.ndarray
            Target labels of shape (n_classes, m_examples)
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
            Updated parameters and epoch loss

        Notes
        -----
        Performs one complete epoch of mini-batch gradient descent training.
        """
        epoch_cost = 0.0
        mini_batches = self.create_mini_batches(X, Y)

        for X_batch, Y_batch in mini_batches:
            AL, caches = forward_propagation_fn(X_batch, parameters)
            epoch_cost += compute_cost_fn(AL, Y_batch)
            gradients = backward_propagation_fn(AL, Y_batch, caches)
            parameters = self.update_parameters(parameters, gradients)

        epoch_cost /= len(mini_batches)
        return parameters, epoch_cost

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        parameters: dict[str, np.ndarray],
        forward_propagation_fn: callable,
        backward_propagation_fn: callable,
        compute_cost_fn: callable,
        epochs: int = 1000,
        print_cost: bool = True,
        print_every: int = 100,
    ) -> dict[str, np.ndarray]:
        """
        Train the model using mini-batch gradient descent.

        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_features, m_examples)
        Y : np.ndarray
            Target labels of shape (n_classes, m_examples)
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

        Notes
        -----
        Trains the model for the specified number of epochs using mini-batch
        gradient descent. Training history is stored in self.history.
        """
        for epoch in range(epochs):
            parameters, epoch_cost = self.train_epoch(
                X,
                Y,
                parameters,
                forward_propagation_fn,
                backward_propagation_fn,
                compute_cost_fn,
            )

            self.history["loss"].append(epoch_cost)

            if print_cost and epoch % print_every == 0:
                print(f"Epoch {epoch}: Cost = {epoch_cost:.6f}")

        return parameters

    def get_config(self) -> dict[str, Any]:
        """
        Get optimizer configuration.

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary
        """
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "optimizer": "MiniBatchGradientDescent",
        }


# Example usage and utility functions
def initialize_parameters(layer_dims: list[int]) -> dict[str, np.ndarray]:
    """
    Initialize parameters for a neural network with given layer dimensions.

    Parameters
    ----------
    layer_dims : List[int]
        List containing the dimensions of each layer

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing initialized parameters
    """
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters[f"W{l}"] = np.random.randn(
            layer_dims[l], layer_dims[l - 1]
        ) * np.sqrt(2.0 / layer_dims[l - 1])
        parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))

    return parameters


def example_usage():
    """Example demonstrating how to use MiniBatchGradientDescent."""
    np.random.seed(42)
    X = np.random.randn(10, 1000)  # 10 features, 1000 examples
    Y = (X[0:1, :] > 0).astype(int)  # Binary classification

    layer_dims = [10, 5, 1]
    parameters = initialize_parameters(layer_dims)

    # Dummy functions (replace with actual implementations)
    def forward_propagation(X, parameters):
        return np.random.randn(1, X.shape[1]), {}

    def backward_propagation(AL, Y, caches):
        return {key: np.random.randn(*val.shape) for key, val in parameters.items()}

    def compute_cost(AL, Y):
        return np.random.rand()

    optimizer = MiniBatchGradientDescent(
        learning_rate=0.01, batch_size=32, shuffle=True
    )

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
    print(f"\nOptimizer config: {optimizer.get_config()}")


if __name__ == "__main__":
    example_usage()
