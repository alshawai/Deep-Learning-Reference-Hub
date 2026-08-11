"""
Optimizer Agreement Tests
=========================

Two implementations of RMSprop used to ship in this subpackage: the canonical one
in ``dlhub.optimizers.rmsprop``, written to be read alongside the derivation, and
the comparison harness's own, written to the uniform driver contract. Two of
Momentum, likewise. They are the same method, so on the same configuration and
the same gradients they must produce the same parameters. They did not.

The cause was narrower than a wrong coefficient. Bias correction is a constructor
flag on the canonical optimizers -- default off for RMSprop, default on for
Momentum -- and the harness copies hardwired it on with no way to ask for
anything else, so a race silently compared bias-corrected RMSprop against what
the rest of the hub calls RMSprop.

The harness now constructs the canonical optimizers and passes the flag through,
so there is one implementation of each method again. These tests are what holds
that true: they are written against a named configuration rather than against
defaults, so they keep their meaning whichever default the race settles on, and
they would fail again the moment either implementation is re-forked.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import numpy as np
import pytest

from dlhub.optimizers import comparison
from dlhub.optimizers.momentum import MomentumOptimizer as CanonicalMomentum
from dlhub.optimizers.rmsprop import RMSpropOptimizer as CanonicalRMSprop

LEARNING_RATE = 0.01
BETA = 0.9
EPSILON = 1e-8
STEPS = 5

# Away from zero in both coordinates and of mixed sign, so a dropped term or a
# sign slip moves the trajectory rather than cancelling out.
START = {"W": np.array([1.0, -2.0]), "b": np.array([0.5])}
GRADIENTS = {"W": np.array([0.3, -0.4]), "b": np.array([0.2])}

METHODS = [
    pytest.param(CanonicalRMSprop, "RMSpropOptimizer", id="rmsprop"),
    pytest.param(CanonicalMomentum, "MomentumOptimizer", id="momentum"),
]


def canonical_trajectory(optimizer, steps=STEPS):
    """Run the canonical interface, which counts its own steps."""
    params = {key: value.copy() for key, value in START.items()}
    trajectory = []
    for _ in range(steps):
        params = optimizer.update_parameters(params, GRADIENTS)
        trajectory.append(params["W"].copy())
    return np.array(trajectory)


def harness_trajectory(optimizer, steps=STEPS):
    """Run the driver interface, which is handed the step index."""
    params = {key: value.copy() for key, value in START.items()}
    trajectory = []
    for t in range(1, steps + 1):
        params = optimizer.update_parameters(params, GRADIENTS, t)
        trajectory.append(params["W"].copy())
    return np.array(trajectory)


@pytest.mark.parametrize("canonical_class, harness_name", METHODS)
class TestTheHarnessAgreesWithTheCanonicalImplementation:
    def test_bias_correction_can_be_turned_off(self, canonical_class, harness_name):
        """
        The configuration the harness could not express while it carried its own
        copies, which applied bias correction unconditionally. It is expressible
        now because the harness constructs the canonical optimizer and passes the
        flag straight through.
        """
        harness_class = getattr(comparison, harness_name)
        harness = harness_class(
            learning_rate=LEARNING_RATE, beta=BETA, bias_correction=False
        )
        canonical = canonical_class(
            learning_rate=LEARNING_RATE, beta=BETA, bias_correction=False
        )
        assert np.allclose(harness_trajectory(harness), canonical_trajectory(canonical))

    def test_bias_correction_can_be_turned_on(self, canonical_class, harness_name):
        """
        The configuration the race runs by default, stated explicitly rather
        than left to a default. Both implementations reach it by being asked, so
        that the race documents what it raced.
        """
        harness_class = getattr(comparison, harness_name)
        harness = harness_class(
            learning_rate=LEARNING_RATE, beta=BETA, bias_correction=True
        )
        canonical = canonical_class(
            learning_rate=LEARNING_RATE, beta=BETA, bias_correction=True
        )
        assert np.allclose(harness_trajectory(harness), canonical_trajectory(canonical))

    def test_the_two_settings_are_not_the_same_run(self, canonical_class, harness_name):
        """
        Guards both tests above from passing vacuously. If the flag were ignored
        rather than honoured, the trajectories would agree either way and the
        agreement tests would prove nothing.
        """
        off = canonical_trajectory(
            canonical_class(
                learning_rate=LEARNING_RATE, beta=BETA, bias_correction=False
            )
        )
        on = canonical_trajectory(
            canonical_class(
                learning_rate=LEARNING_RATE, beta=BETA, bias_correction=True
            )
        )
        assert not np.allclose(off, on)


def test_the_race_default_differs_from_the_canonical_rmsprop_default():
    """
    What is left of the divergence once the fork is gone, and why it is now a
    property rather than a defect. The race asks for bias correction everywhere
    it is available; canonical RMSprop, constructed on its own, defaults to
    none. Those are different runs, by a margin far larger than a floating-point
    tolerance.

    The difference is the same size as before. What changed is that it is now a
    stated configuration rather than a hardwired one -- ``RACE_BIAS_CORRECTION``
    names it, and the tests above show either setting can be asked for and is
    honoured by both implementations. This test pins that the race's choice is a
    real choice: if the constant flipped to False, the race would silently
    become an uncorrected run and this assertion would catch it.
    """
    canonical = canonical_trajectory(
        CanonicalRMSprop(learning_rate=LEARNING_RATE, beta=BETA, epsilon=EPSILON)
    )
    harness = harness_trajectory(
        comparison.RMSpropOptimizer(
            learning_rate=LEARNING_RATE, beta=BETA, epsilon=EPSILON
        )
    )
    assert not np.allclose(harness, canonical), (
        "the race and the canonical default now agree; RACE_BIAS_CORRECTION has "
        "been flipped off, or one of the two defaults moved"
    )
