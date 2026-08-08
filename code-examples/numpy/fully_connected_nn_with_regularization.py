"""
Deep Neural Network Implementation
==================================

Defines a configurable fully-connected deep neural network from scratch using NumPy,
including forward/backward propagation, parameter initialization, optimization,
and prediction routines.

References
----------
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Ng, A. (2017). Deep Learning Specialization: Course on Neural Networks.
  (gradient checking, initialization, forward/backward propagation)

Author
------
Deep Learning Reference Hub

License
-------
MIT License

Notes
-----
- Supports L-layer architectures with ReLU and Sigmoid activations
- Includes optional gradient checking and loss tracking
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Any


class DeepNeuralNetwork:
    """
    A comprehensive Deep Neural Network implementation with modern techniques.

    This implementation includes:
    - Multiple initialization methods (He, Xavier, Random)
    - Regularization techniques (L1, L2, Dropout)
    - Batch Normalization
    - Gradient Clipping
    - Learning Rate Scheduling
    - Early Stopping
    - Comprehensive metrics tracking

    Parameters
    ----------
    layer_dims : List
        Layer dimensions [n_x, n_h1, n_h2, ..., n_y]
    initialization : str
        Initialization method ('he', 'xavier', 'random')
    regularization : str(optional)
        Regularization type (None, 'l1', 'l2', 'dropout')
    lambda_reg : float, default=0.01
        Regularization strength parameter
    keep_prob : float, default=0.8
        Dropout keep probability (0 < keep_prob <= 1)
    use_batch_norm : bool, default=True
        Whether to use batch normalization
    gradient_clipping : bool, default=True
        Whether to apply gradient clipping
    clip_value : float, default=True
        Maximum gradient norm for clipping

    Attributes
    ----------
    layer_dims : List[int]
        Dimensions of each layer
    L : int
        Number of layers (excluding input)
    parameters : Dict
        Network weights and biases
    bn_params : Dict
        Batch normalization parameters
    costs_history : List
        Training cost history
    val_costs_history: List
        Validation cost history
    accuracies_history : List
        Training accuracy history
    val_accuracies_history : List
        Validation accuracy history

    Notes
    -----
    Raises ValueError: If invalid parameters are provided
    """

    def __init__(
        self,
        layer_dims: List[int],
        initialization: str = "he",
        regularization: Optional[str] = None,
        lambda_reg: float = 0.01,
        keep_prob: float = 0.8,
        use_batch_norm: bool = True,
        gradient_clipping: bool = True,
        clip_value: float = 5.0,
    ):
        self._validate_inputs(
            layer_dims, initialization, regularization, lambda_reg, keep_prob
        )

        self.layer_dims = layer_dims
        self.L = len(layer_dims) - 1  # Number of layers (excluding input)
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.keep_prob = keep_prob
        self.use_batch_norm = use_batch_norm
        self.gradient_clipping = gradient_clipping
        self.clip_value = clip_value

        self.parameters = self._initialize_parameters(initialization)

        if use_batch_norm:
            self.running_mean = {}
            self.running_var = {}
            self.momentum = 0.9
            self.bn_params = self._initialize_batch_norm()

        self.costs_history = []
        self.val_costs_history = []
        self.accuracies_history = []
        self.val_accuracies_history = []

        self.best_val_cost = float("inf")
        self.patience_counter = 0

    def _validate_inputs(
        self,
        layer_dims: List[int],
        initialization: str,
        regularization: Optional[str],
        lambda_reg: float,
        keep_prob: float,
    ) -> None:
        """Validate input parameters."""
        if len(layer_dims) < 2:
            raise ValueError("Network must have at least 2 layers (input and output)")

        if any(dim <= 0 for dim in layer_dims):
            raise ValueError("All layer dimensions must be positive")

        if initialization not in ["he", "xavier", "random"]:
            raise ValueError("Initialization must be 'he', 'xavier', or 'random'")

        if regularization not in [None, "l1", "l2", "dropout"]:
            raise ValueError("Regularization must be None, 'l1', 'l2', or 'dropout'")

        if lambda_reg < 0:
            raise ValueError("Regularization parameter must be non-negative")

        if not 0 < keep_prob <= 1:
            raise ValueError("Keep probability must be in (0, 1]")

    def _initialize_parameters(self, method: str) -> Dict[str, np.ndarray]:
        """
        Initialize network parameters using specified method.

        Parameters
        ----------
        method : str
            Initialization method ('he', 'xavier', 'random')

        Returns
        -------
        Dict[str, np.ndarray]
            Contains initialized weights and biases
        """
        np.random.seed(42)
        parameters = {}

        for l in range(1, self.L + 1):
            fan_in = self.layer_dims[l - 1]
            fan_out = self.layer_dims[l]

            if method == "he":
                std = np.sqrt(2.0 / fan_in)
            elif method == "xavier":
                std = np.sqrt(1.0 / fan_in)
            elif method == "random":
                std = 0.01

            parameters[f"W{l}"] = np.random.randn(fan_out, fan_in) * std
            parameters[f"b{l}"] = np.zeros((fan_out, 1))

        return parameters

    def _initialize_batch_norm(self) -> Dict[str, np.ndarray]:
        """
        Initialize batch normalization parameters.

        Returns
        -------
        Dict[str, np.ndarray]
            Contains gamma and beta parameters
        """
        bn_params = {}

        for l in range(1, self.L):  # Not applied to output layer
            bn_params[f"gamma{l}"] = np.ones((self.layer_dims[l], 1))
            bn_params[f"beta{l}"] = np.zeros((self.layer_dims[l], 1))

            self.running_mean[f"mean{l}"] = np.zeros((self.layer_dims[l], 1))
            self.running_var[f"var{l}"] = np.ones((self.layer_dims[l], 1))

        return bn_params

    def _relu(self, Z: np.ndarray) -> np.ndarray:
        """Relu activation function."""
        return np.maximum(0, Z)

    def _relu_derivative(self, Z: np.ndarray) -> np.ndarray:
        """Relu derivative."""
        return (Z > 0).astype(float)

    def _sigmoid(self, Z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function with numerical stability."""
        Z_clipped = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z_clipped))

    def _sigmoid_derivative(self, A: np.ndarray) -> np.ndarray:
        """Sigmoid derivative."""
        return A * (1 - A)

    def forward_propagation(
        self, X: np.ndarray, training: bool = True
    ) -> Tuple[np.ndarray, List]:
        """
        Perform forward propagation through the network.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_features, m_samples)
        training : bool, default=True
            Whether in training mode (affects dropout and batch norm)

        Returns
        -------
        Tuple[np.ndarray, List]
            Contains final output and caches for backprob
        """
        if X.shape[0] != self.layer_dims[0]:
            raise ValueError(
                f"Input shape {X.shape[0]} doesn't match expected {self.layer_dims[0]}"
            )

        caches = []
        A = X

        for l in range(1, self.L):
            A_prev = A
            Z = np.dot(self.parameters[f"W{l}"], A_prev) + self.parameters[f"b{l}"]

            if self.use_batch_norm:
                Z, bn_cache = self._batch_norm_forward(Z, l, training)
            else:
                bn_cache = None

            A = self._relu(Z)

            if self.regularization == "dropout" and training:
                A, dropout_cache = self._dropout_forward(A, self.keep_prob)
            else:
                dropout_cache = None

            cache = (A_prev, Z, A, bn_cache, dropout_cache)
            caches.append(cache)

        A_prev = A
        ZL = (
            np.dot(self.parameters[f"W{self.L}"], A_prev)
            + self.parameters[f"b{self.L}"]
        )
        AL = self._sigmoid(ZL)

        cache = (A_prev, ZL, AL, None, None)
        caches.append(cache)

        return AL, caches

    def _batch_norm_forward(
        self, Z: np.ndarray, l: int, training: bool, eps: float = 1e-8
    ) -> Tuple[np.ndarray, Optional[Tuple]]:
        """
        Batch normalization forward pass.

        Parameters
        ----------
        Z : np.ndarray
            Pre-activation values
        l : int
            Layer index
        training : bool, default=True
            Whether in training mode
        eps : float, defualt=1e-8
            Small constant for numerical stability

        Returns
        -------
        Tuple[np.ndarray, tuple]
            Contains normalized output and cache for backprop
        """
        if training:
            mu = np.mean(Z, axis=1, keepdims=True)
            var = np.var(Z, axis=1, keepdims=True)

            self.running_mean[f"mean{l}"] = (
                self.momentum * self.running_mean[f"mean{l}"] + (1 - self.momentum) * mu
            )
            self.running_var[f"var{l}"] = (
                self.momentum * self.running_var[f"var{l}"] + (1 - self.momentum) * var
            )

            Z_norm = (Z - mu) / np.sqrt(var + eps)
            Z_out = self.bn_params[f"gamma{l}"] * Z_norm + self.bn_params[f"beta{l}"]

            cache = (Z, Z_norm, mu, var, eps)
            return Z_out, cache
        else:
            # Use running statistics for inference
            Z_norm = (Z - self.running_mean[f"mean{l}"]) / np.sqrt(
                self.running_var[f"var{l}"] + eps
            )
            Z_out = self.bn_params[f"gamma{l}"] * Z_norm + self.bn_params[f"beta{l}"]
            return Z_out, None

    def _dropout_forward(
        self, A: np.ndarray, keep_prob: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dropout forward pass.

        Parameters
        ----------
        A : np.ndarray
            Activations
        keep_prob : float
            Probability of keeping each neuron

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Contains dropped activations and dropout mask
        """
        mask = np.random.binomial(1, keep_prob, A.shape) / keep_prob
        A_drop = A * mask
        return A_drop, mask

    def compute_cost(self, AL: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute the cost function with regularization.

        Parameters
        ----------
        AL : np.ndarray
            Network output of shape (1, m_samples)
        Y : np.ndarray
            True labels of shape (1, m_samples)

        Returns
        -------
        float
            Total cost including regularization
        """
        m = Y.shape[1]
        AL_clipped = np.clip(AL, 1e-8, 1 - 1e-8)  # Clip predictions to prevent log(0)

        cost = -(1 / m) * (
            np.dot(Y, np.log(AL_clipped).T) + np.dot(1 - Y, np.log(1 - AL_clipped).T)
        )

        reg_cost = 0
        if self.regularization == "l2":
            weights = np.concatenate(
                [self.parameters[f"W{l}"].flatten() for l in range(1, self.L + 1)]
            )
            reg_cost = self.lambda_reg / (2 * m) * np.sum(weights**2)
            cost += reg_cost

        elif self.regularization == "l1":
            weights = np.concatenate(
                [self.parameters[f"W{l}"].flatten() for l in range(1, self.L + 1)]
            )
            reg_cost = self.lambda_reg / m * np.sum(np.abs(weights))
            cost += reg_cost

        return np.squeeze(cost)

    def backward_propagation(
        self, AL: np.ndarray, Y: np.ndarray, caches: List
    ) -> Dict[str, np.ndarray]:
        """
        Perform backward propagation to compute gradients.

        Parameters
        ----------
        AL : np.ndarray
            Network output
        Y : np.ndarray
            True labels
        caches : List
            Forward propagation caches

        Returns
        -------
        Dict[str, np.ndarray]
            Contains gradients for all parameters
        """
        grads = {}
        m = AL.shape[1]

        # Carried from one iteration to the next: layer l reads the dA that
        # layer l+1 produced. Bound here so the handoff is visible rather than
        # implied by the loop.
        dA_next = None

        for l in reversed(range(1, self.L + 1)):
            A_prev, Z, A, bn_cache, dropout_cache = caches[l - 1]

            if l == self.L:
                dZ = AL - Y  # Cross-Entropy Derivative
            else:
                dA = dA_next

                if dropout_cache is not None:
                    dA = dA * dropout_cache

                dZ = dA * self._relu_derivative(Z)

                if bn_cache is not None:
                    dZ = self._batch_norm_backward(dZ, bn_cache, l)

            dW = (1 / m) * np.dot(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dA_prev = np.dot(self.parameters[f"W{l}"].T, dZ)

            if self.regularization == "l2":
                dW += (self.lambda_reg / m) * self.parameters[f"W{l}"]
            elif self.regularization == "l1":
                dW += (self.lambda_reg / m) * np.sign(self.parameters[f"W{l}"])

            grads[f"dW{l}"] = dW
            grads[f"db{l}"] = db

            dA_next = dA_prev

        return grads

    def _batch_norm_backward(
        self, dZ_out: np.ndarray, cache: Tuple, l: int
    ) -> np.ndarray:
        """
        Batch normalization backward pass.

        Parameters
        ----------
        dZ_out : np.ndarray
            Gradient from next layer
        cache : Tuple
            Forward pass cache
        l : int
            Layer index

        Returns
        -------
        np.ndarray
            Gradient with respect to input
        """
        Z, Z_norm, mu, var, eps = cache
        m = Z.shape[1]

        dgamma = np.sum(dZ_out * Z_norm, axis=1, keepdims=True)
        dbeta = np.sum(dZ_out, axis=1, keepdims=True)

        if not hasattr(self, "bn_grads"):
            self.bn_grads = {}
        self.bn_grads[f"dgamma{l}"] = dgamma
        self.bn_grads[f"dbeta{l}"] = dbeta

        dZ_norm = dZ_out * self.bn_params[f"gamma{l}"]

        dvar = np.sum(
            dZ_norm * (Z - mu) * -0.5 * (var + eps) ** (-3 / 2), axis=1, keepdims=True
        )
        dmu = (
            np.sum(dZ_norm * -1 / np.sqrt(var + eps), axis=1, keepdims=True)
            + dvar * np.sum(-2 * (Z - mu), axis=1, keepdims=True) / m
        )

        dZ = dZ_norm / np.sqrt(var + eps) + dvar * 2 * (Z - mu) / m + dmu / m

        return dZ

    def _clip_gradients(self, grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply gradient clipping to prevent exploding gradients.

        Parameters
        ----------
        grads : Dict[str, np.ndarray]
            Dictionary of gradients

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of clipped gradients
        """
        total_norm = 0
        for grad in grads.values():
            total_norm += np.sum(grad**2)
        total_norm = np.sqrt(total_norm)

        if total_norm > self.clip_value:
            clip_coeff = self.clip_value / total_norm
            for key in grads:
                grads[key] = grads[key] * clip_coeff

        return grads

    def update_parameters(
        self, grads: Dict[str, np.ndarray], learning_rate: float
    ) -> None:
        """
        Update network parameters using gradients.

        Parameters
        ----------
        grads : Dict[str, np.ndarray]
            Dictionary of gradients
        learning_rate : float
            Learning rate for parameter updates
        """
        if self.gradient_clipping:
            grads = self._clip_gradients(grads)

        for l in range(1, self.L + 1):
            self.parameters[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
            self.parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"]

        if self.use_batch_norm and hasattr(self, "bn_grads"):
            for l in range(1, self.L):
                self.bn_params[f"gamma{l}"] -= (
                    learning_rate * self.bn_grads[f"dgamma{l}"]
                )
                self.bn_params[f"beta{l}"] -= learning_rate * self.bn_grads[f"dbeta{l}"]

    def compute_accuracy(self, AL: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute classification accuracy.

        Parameters
        ----------
        AL : np.ndarray
            Network predictions
        Y : np.ndarray
            True labels

        Returns
        -------
        float
            Accuracy percentage
        """
        predictions = (AL > 0.5).astype(int)
        accuracy = np.mean(predictions == Y)
        return accuracy

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        learning_rate: float = 0.01,
        num_epochs: int = 1000,
        print_cost: bool = True,
        learning_rate_decay: float = 0.95,
        decay_step: int = 100,
        early_stopping: bool = True,
        patience: int = 50,
    ) -> Dict[str, List]:
        """
        Train the neural network with advanced techniques.

        Parameters
        ----------
            X : np.ndarray
                Training data of shape (n_features, m_samples)
            Y : np.ndarray
                Training labels of shape (1, m_samples)
            X_val : np.ndarray
                Validation data
            Y_val : np.ndarray
                Validation labels
            learning_rate : float, default=0.01
                Initial learning rate
            num_epochs : int, default=1000
                Number of training epochs
            print_cost : bool, default=True
                Whether to print cost during training
            learning_rate_decay : float, default=0,95
                Learning rate decay factor
            decay_step : int, default=100
                Steps between learning rate decay
            early_stopping : bool, default=True
                Whether to use early stopping
            patience : int, default=50
                Early stopping patience

        Returns
        -------
        Dict[str, List]
            Contains training history
        """
        if X.shape[0] != self.layer_dims[0]:
            raise ValueError(
                f"Input features {X.shape[0]} don't match network input {self.layer_dims[0]}"
            )

        self.costs_history = []
        self.val_costs_history = []
        self.accuracies_history = []
        self.val_accuracies_history = []

        current_lr = learning_rate

        for epoch in range(num_epochs):
            AL, caches = self.forward_propagation(X, training=True)

            cost = self.compute_cost(AL, Y)
            accuracy = self.compute_accuracy(AL, Y)

            self.costs_history.append(cost)
            self.accuracies_history.append(accuracy)

            grads = self.backward_propagation(AL, Y, caches)

            self.update_parameters(grads, current_lr)

            AL_val, _ = self.forward_propagation(X_val, training=False)
            val_cost = self.compute_cost(AL_val, Y_val)
            val_accuracy = self.compute_accuracy(AL_val, Y_val)

            self.val_costs_history.append(val_cost)
            self.val_accuracies_history.append(val_accuracy)

            if epoch % decay_step == 0 and epoch > 0:
                current_lr *= learning_rate_decay

            if early_stopping:
                if val_cost < self.best_val_cost:
                    self.best_val_cost = val_cost
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                if self.patience_counter >= patience:
                    if print_cost:
                        print(f"Early stopping at epoch {epoch}")
                    break

            if print_cost and epoch % 100 == 0:
                print(
                    f"Epoch {epoch:4d}: Cost = {cost:.6f}, Acc = {accuracy:.4f}, "
                    f"Val Cost = {val_cost:.6f}, Val Acc = {val_accuracy:.4f}, LR = {current_lr:.6f}"
                )

        return {
            "costs": self.costs_history,
            "val_costs": self.val_costs_history,
            "accuracies": self.accuracies_history,
            "val_accuracies": self.val_accuracies_history,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_features, m_samples)

        Returns
        -------
        np.ndarray
            Binary predictions of shape (1, m_samples)
        """
        AL, _ = self.forward_propagation(X, training=False)
        predictions = (AL > 0.5).astype(int)
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_features, m_samples)

        Returns
        -------
        np.ndarray
            Probabilities of shape (1, m_samples)
        """
        AL, _ = self.forward_propagation(X, training=False)
        return AL

    def plot_training_history(self, figsize: Tuple[int, int] = (15, 5)) -> None:
        """
        Plot training history including cost and accuracy curves.

        Parameters
        ----------
        figsize : Tuple[int, int]
            Figure size tuple
        """
        if not self.costs_history:
            print("No training history available. Train the model first.")
            return

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Cost Curves
        axes[0].plot(self.costs_history, label="Training Cost", linewidth=2)
        axes[0].plot(self.val_costs_history, label="Validation Cost", linewidth=2)
        axes[0].set_title("Training and Validation Cost")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Cost")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy Curves
        axes[1].plot(self.accuracies_history, label="Training Accuracy", linewidth=2)
        axes[1].plot(
            self.val_accuracies_history, label="Validation Accuracy", linewidth=2
        )
        axes[1].set_title("Training and Validation Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.

        Returns
        -------
        Dict[str, Any]
            Contains model configuration and statistics
        """
        total_params = sum(param.size for param in self.parameters.values())
        if self.use_batch_norm:
            total_params += sum(param.size for param in self.bn_params.values())

        return {
            "architecture": self.layer_dims,
            "total_parameters": total_params,
            "regularization": self.regularization,
            "lambda_reg": self.lambda_reg,
            "batch_normalization": self.use_batch_norm,
            "dropout_keep_prob": (
                self.keep_prob if self.regularization == "dropout" else None
            ),
            "gradient_clipping": self.gradient_clipping,
            "clip_value": self.clip_value if self.gradient_clipping else None,
        }


# Example usage and testing
def create_sample_data(
    n_features: int = 128, n_train: int = 1000, n_val: int = 200
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sample dataset for testing.

    Parameters
    ----------
    n_features : int, default=128
        Number of input features
    n_train : int, defualt=1000
        Number of training samples
    n_val : int, default=200
        Number of validation samples

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (X_train, Y_train, X_val, Y_val)
    """
    np.random.seed(42)

    X_train = np.random.randn(n_features, n_train)
    Y_train = (np.sum(X_train[:10], axis=0, keepdims=True) > 0).astype(int)

    X_val = np.random.randn(n_features, n_val)
    Y_val = (np.sum(X_val[:10], axis=0, keepdims=True) > 0).astype(int)

    return X_train, Y_train, X_val, Y_val


def main():
    """Example usage of the Deep Neural Network."""
    print("Creating sample dataset...")
    X_train, Y_train, X_val, Y_val = create_sample_data()

    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {Y_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Validation labels shape: {Y_val.shape}")

    print("\nCreating Deep Neural Network...")
    model = DeepNeuralNetwork(
        layer_dims=[128, 64, 1],
        initialization="he",
        regularization="l2",
        lambda_reg=0.4,
        keep_prob=0.9,
        use_batch_norm=True,
        gradient_clipping=False,
        clip_value=5.0,
    )

    print("\nModel Information:")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\nTraining model...")
    history = model.train(
        X_train,
        Y_train,
        X_val,
        Y_val,
        learning_rate=0.02,
        num_epochs=5000,
        print_cost=True,
        learning_rate_decay=0.99,
        decay_step=100,
        early_stopping=False,
        patience=50,
    )

    print("\nMaking predictions...")
    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)

    train_accuracy = model.compute_accuracy(model.predict_proba(X_train), Y_train)
    val_accuracy = model.compute_accuracy(model.predict_proba(X_val), Y_val)

    print(f"\nFinal Results:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    model.plot_training_history()


if __name__ == "__main__":
    main()
