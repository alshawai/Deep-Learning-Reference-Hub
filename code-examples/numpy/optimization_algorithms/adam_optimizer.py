"""
Adam Optimizer Implementation
============================

A comprehensive implementation of the Adam (Adaptive Moment Estimation) optimizer
with bias correction, gradient clipping, and numerical stability features.

References
----------
- Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.
  https://arxiv.org/abs/1412.6980

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import warnings

import numpy as np


class AdamOptimizer:
    """
    Adam (Adaptive Moment Estimation) Optimizer

    Adam combines the advantages of AdaGrad and RMSProp by computing adaptive
    learning rates for each parameter using estimates of first and second
    moments of the gradients.

    The algorithm maintains exponentially decaying averages of past gradients
    and past squared gradients, which act as estimates of the first moment
    (mean) and second moment (uncentered variance) of the gradients.

    Parameters
    ----------
    learning_rate : float, default=0.001
        Learning rate (alpha in the paper)
    beta1 : float, default=0.9
        Exponential decay rate for first moment estimates
    beta2 : float, default=0.999
        Exponential decay rate for second moment estimates
    epsilon : float, default=1e-8
        Small constant for numerical stability
    weight_decay : float, default=0.0
        Weight decay coefficient (L2 regularization)
    amsgrad : bool, default=False
        Whether to use AMSGrad variant which maintains maximum of squared gradients
    gradient_clip_norm : float, optional
        Maximum norm for gradient clipping
    gradient_clip_value : float, optional
        Maximum absolute value for gradient clipping

    Attributes
    ----------
    m : dict
        First moment estimates (exponentially decaying average of gradients)
    v : dict
        Second moment estimates (exponentially decaying average of squared gradients)
    v_hat_max : dict
        Maximum of v_hat values (used in AMSGrad)
    t : int
        Time step (number of updates performed)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        gradient_clip_norm: float | None = None,
        gradient_clip_value: float | None = None,
    ):
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError(f"Invalid learning rate: {learning_rate}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {beta2}")
        if epsilon <= 0.0:
            raise ValueError(f"Invalid epsilon value: {epsilon}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_clip_value = gradient_clip_value

        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}
        self.v_hat_max: dict[str, np.ndarray] = {}
        self.t = 0

        self.history = {
            "loss": [],
            "gradient_norm": [],
            "parameter_norm": [],
            "learning_rate": [],
        }

    def _initialize_moments(
        self, param_name: str, param_shape: tuple[int, ...]
    ) -> None:
        """Initialize moment estimates for a parameter."""
        if param_name not in self.m:
            self.m[param_name] = np.zeros(param_shape, dtype=np.float64)
            self.v[param_name] = np.zeros(param_shape, dtype=np.float64)
            if self.amsgrad:
                self.v_hat_max[param_name] = np.zeros(param_shape, dtype=np.float64)

    def _clip_gradients(
        self, gradients: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """
        Apply gradient clipping if specified.

        Parameters
        ----------
        gradients : dict
            Dictionary of gradients for each parameter

        Returns
        -------
        dict
            Clipped gradients
        """
        if self.gradient_clip_norm is not None:
            total_norm = 0.0
            for grad in gradients.values():
                total_norm += np.sum(grad**2)
            total_norm = np.sqrt(total_norm)

            if total_norm > self.gradient_clip_norm:
                clip_coeff = self.gradient_clip_norm / (total_norm + 1e-8)
                gradients = {
                    name: grad * clip_coeff for name, grad in gradients.items()
                }

        if self.gradient_clip_value is not None:
            gradients = {
                name: np.clip(grad, -self.gradient_clip_value, self.gradient_clip_value)
                for name, grad in gradients.items()
            }

        return gradients

    def update(
        self, parameters: dict[str, np.ndarray], gradients: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """
        Perform a single optimization step.

        Parameters
        ----------
        parameters : dict
            Dictionary of parameters to optimize
        gradients : dict
            Dictionary of gradients for each parameter

        Returns
        -------
        dict
            Updated parameters
        """
        self.t += 1
        gradients = self._clip_gradients(gradients)

        grad_norm = np.sqrt(sum(np.sum(grad**2) for grad in gradients.values()))
        param_norm = np.sqrt(sum(np.sum(param**2) for param in parameters.values()))

        updated_parameters = {}
        for param_name, param in parameters.items():
            if param_name not in gradients:
                updated_parameters[param_name] = param.copy()
                continue

            grad = gradients[param_name]

            self._initialize_moments(param_name, param.shape)

            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param

            self.m[param_name] = (
                self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
            )
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (
                grad**2
            )
            m_hat = self.m[param_name] / (1 - self.beta1**self.t)
            v_hat = self.v[param_name] / (1 - self.beta2**self.t)

            # AMSGrad modification
            if self.amsgrad:
                self.v_hat_max[param_name] = np.maximum(
                    self.v_hat_max[param_name], v_hat
                )
                v_hat = self.v_hat_max[param_name]

            denominator = np.sqrt(v_hat) + self.epsilon
            step = self.learning_rate * m_hat / denominator

            if np.any(np.isnan(step)) or np.any(np.isinf(step)):
                warnings.warn(
                    f"Numerical instability detected in parameter {param_name}"
                )
                step = np.nan_to_num(step, nan=0.0, posinf=1e-6, neginf=-1e-6)

            updated_parameters[param_name] = param - step

        self.history["gradient_norm"].append(grad_norm)
        self.history["parameter_norm"].append(param_norm)
        self.history["learning_rate"].append(self.learning_rate)

        return updated_parameters

    def get_config(self) -> dict:
        """
        Get optimizer configuration.

        Returns
        -------
        dict
            Configuration dictionary
        """
        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "weight_decay": self.weight_decay,
            "amsgrad": self.amsgrad,
            "gradient_clip_norm": self.gradient_clip_norm,
            "gradient_clip_value": self.gradient_clip_value,
            "time_step": self.t,
        }

    def reset_state(self) -> None:
        """Reset optimizer state (moments and time step)."""
        self.m.clear()
        self.v.clear()
        self.v_hat_max.clear()
        self.t = 0
        self.history = {
            "loss": [],
            "gradient_norm": [],
            "parameter_norm": [],
            "learning_rate": [],
        }

    def get_state(self) -> dict:
        """
        Get complete optimizer state.

        Returns
        -------
        dict
            Complete state dictionary
        """
        return {
            "config": self.get_config(),
            "moments": {
                "m": self.m.copy(),
                "v": self.v.copy(),
                "v_hat_max": self.v_hat_max.copy() if self.amsgrad else {},
            },
            "history": self.history.copy(),
        }

    def load_state(self, state: dict) -> None:
        """
        Load optimizer state.

        Parameters
        ----------
        state : dict
            State dictionary from get_state()
        """
        config = state["config"]
        self.learning_rate = config["learning_rate"]
        self.beta1 = config["beta1"]
        self.beta2 = config["beta2"]
        self.epsilon = config["epsilon"]
        self.weight_decay = config["weight_decay"]
        self.amsgrad = config["amsgrad"]
        self.gradient_clip_norm = config["gradient_clip_norm"]
        self.gradient_clip_value = config["gradient_clip_value"]
        self.t = config["time_step"]

        moments = state["moments"]
        self.m = moments["m"].copy()
        self.v = moments["v"].copy()
        self.v_hat_max = moments["v_hat_max"].copy()

        self.history = state["history"].copy()


def create_adam_optimizer(
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    **kwargs,
) -> AdamOptimizer:
    """
    Factory function to create Adam optimizer with common configurations.

    Parameters
    ----------
    learning_rate : float
        Learning rate
    beta1 : float
        First moment decay rate
    beta2 : float
        Second moment decay rate
    epsilon : float
        Numerical stability constant
    **kwargs
        Additional optimizer parameters

    Returns
    -------
    AdamOptimizer
        Configured Adam optimizer
    """
    return AdamOptimizer(
        learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon, **kwargs
    )


if __name__ == "__main__":
    np.random.seed(42)

    parameters = {
        "W1": np.random.randn(3, 4) * 0.1,
        "b1": np.zeros((3, 1)),
        "W2": np.random.randn(1, 3) * 0.1,
        "b2": np.zeros((1, 1)),
    }

    optimizer = AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
    print("Adam Optimizer Test")
    print("=" * 50)
    print(f"Initial configuration: {optimizer.get_config()}")

    for step in range(5):
        # Random gradients (need to get them from backpropagation)
        gradients = {
            name: np.random.randn(*param.shape) * 0.01
            for name, param in parameters.items()
        }

        parameters = optimizer.update(parameters, gradients)

        print(f"\nStep {step + 1}:")
        print(f"  Gradient norm: {optimizer.history['gradient_norm'][-1]:.6f}")
        print(f"  Parameter norm: {optimizer.history['parameter_norm'][-1]:.6f}")

    print(
        f"\nFinal optimizer state after {optimizer.t} steps:\n{optimizer.get_state()}"
    )
    print("\nOptimization completed successfully!")
