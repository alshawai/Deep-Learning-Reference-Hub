"""
Optimizer Agreement Tests
=========================

Two implementations of RMSprop ship in this subpackage: the canonical one in
``dlhub.optimizers.rmsprop``, written to be read alongside the derivation, and
the comparison harness's own, written to the uniform driver contract. Two of
Momentum, likewise. They are the same method, so on the same configuration and
the same gradients they must produce the same parameters.

They do not, and the reason is not a coefficient. Bias correction is a
constructor flag on the canonical optimizers -- default off for RMSprop, default
on for Momentum -- and the harness copies hardwire it on with no way to ask for
anything else. So a race run through the harness silently compares
bias-corrected RMSprop against whatever the rest of the hub means by RMSprop,
and the module docstring's claim that "all optimizers include proper bias
correction" is true only of the copies.

The tests below are written against the configuration, not the defaults: each
one names the bias-correction setting it wants and requires both
implementations to honour it. That is the property that has to survive the fix,
whichever default the harness ends up choosing.

This file ships ahead of the commit that rewires the harness onto the canonical
optimizers, so the two agreement tests ship failing, deliberately. The failure
is the divergence itself, demonstrated rather than asserted in a plan. It is
recorded as a strict ``xfail`` on the exact exception the missing flag raises,
which means the suite stays honest in both directions: red is expected now, and
the moment the rewiring makes either test pass, the strict marker turns that
unexpected pass into a failure until the marker is removed. The next commit
removes them.

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
    @pytest.mark.xfail(
        strict=True,
        raises=TypeError,
        reason="the harness carries its own copies, which hardwire bias correction "
        "on and take no flag; rewiring it onto the canonical optimizers is what "
        "makes this expressible",
    )
    def test_bias_correction_can_be_turned_off(self, canonical_class, harness_name):
        """
        The configuration the harness cannot currently express. Its copies apply
        bias correction unconditionally, so this is the assertion that fails
        until the harness runs the canonical optimizer.
        """
        harness_class = getattr(comparison, harness_name)
        harness = harness_class(
            learning_rate=LEARNING_RATE, beta=BETA, bias_correction=False
        )
        canonical = canonical_class(
            learning_rate=LEARNING_RATE, beta=BETA, bias_correction=False
        )
        assert np.allclose(harness_trajectory(harness), canonical_trajectory(canonical))

    @pytest.mark.xfail(
        strict=True,
        raises=TypeError,
        reason="same missing flag: the harness reaches this configuration only by "
        "having no alternative, so it cannot be asked for it",
    )
    def test_bias_correction_can_be_turned_on(self, canonical_class, harness_name):
        """
        The configuration the harness already runs, stated explicitly rather
        than left to a default. Both implementations must reach it by being
        asked, so that the race documents what it raced.
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


def test_the_harness_rmsprop_is_reachable_under_the_canonical_default():
    """
    The divergence stated as the number it is worth. Canonical RMSprop defaults
    to no bias correction; the harness copy has no such setting, and the gap
    between the two trajectories is far larger than a floating-point tolerance,
    so a reader comparing the harness's curve against the canonical module's
    output sees two different optimizers under one name.
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
        "the defaults agree; this test and the divergence it records are stale"
    )
