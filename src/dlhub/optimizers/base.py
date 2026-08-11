"""
Optimizer Contract
==================

The interface every optimizer in this subpackage presents to code that drives a
training loop: given the current parameters and their gradients, return the
updated parameters.

The base class is deliberately thin. An optimizer is the artifact a reader came
to read, so the update rule stays written out in full in its own module rather
than being assembled from hooks defined here. What this class supplies is the
uniform signature that lets a driver hold a collection of optimizers without
knowing which one it has -- the comparison harness being the case that motivated
extracting it.

Two conventions worth stating, because they are the ones a new optimizer gets
wrong:

Bias correction is the optimizer's own decision, not the contract's. Momentum and
RMSprop maintain running averages that start at zero and are therefore biased
toward zero for the first few steps; whether to divide that bias out is a property
of the method, and the modules that implement those methods expose it as a
constructor flag. A driver that wants a like-for-like race across optimizers has
to set that flag deliberately rather than inherit whatever each default happens
to be.

The step index is passed in. Optimizers whose update depends on how many steps
have been taken -- anything applying bias correction -- read it from the argument
rather than counting internally, so that a driver resetting an optimizer between
runs does not have to trust it to reset its own counter. The canonical optimizer
modules in this subpackage predate this contract and count internally instead; the
adapters in ``comparison`` bridge the two by clearing that counter in ``reset``,
which is the property the contract actually cares about and the one its tests
check.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import numpy as np


class BaseOptimizer:
    """
    Base class for all optimizers with common functionality.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate for parameter updates.

    Attributes
    ----------
    learning_rate : float
        Step size applied to each update.
    name : str
        Human-readable label, used to key and plot results. Subclasses set it to
        the name of the method they implement.
    """

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.name = "BaseOptimizer"

    def update_parameters(
        self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray], t: int
    ) -> dict[str, np.ndarray]:
        """
        Update parameters using optimization algorithm.

        Implementations return a new dictionary rather than mutating the one they
        were given, so that a caller can keep the parameter trajectory of a run.

        Parameters
        ----------
        params : dict
            Current parameter values.
        grads : dict
            Gradients for each parameter.
        t : int
            Current iteration, counted from one. Optimizers applying bias
            correction divide by ``1 - beta ** t``, which is why the count starts
            at one rather than zero.

        Returns
        -------
        dict
            Updated parameters.

        Raises
        ------
        NotImplementedError
            Always, on the base class. An optimizer is defined by its update
            rule, so there is no meaningful default to inherit.
        """
        raise NotImplementedError("Subclasses must implement update_parameters")

    def reset(self) -> None:
        """
        Reset optimizer state for new optimization run.

        The base implementation does nothing, which is correct for a stateless
        optimizer such as plain gradient descent. Any optimizer accumulating
        state across steps -- a velocity, a second moment -- must override this
        and clear it, or a second run starts from wherever the first one ended
        and its trajectory is not reproducible.
        """
        pass
