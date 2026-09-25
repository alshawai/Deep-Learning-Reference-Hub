"""
Minimal Gated Cell Reference Tests
==================================

The correctness proof for minGRU and minLSTM. This module ships one cell in
three forms -- the serial recurrence, the plain-space associative scan, and the
log-space scan -- and the failure worth catching is that they quietly stop being
the same function. A scan that assumes a power-of-two length, an association
order that drops a term, or a log-space rearrangement with a sign the wrong way
round all still return an array of the right shape, full of plausible numbers.
The equivalence tests therefore pin all three against each other on the
dossier's fixture, at its ``atol=1e-10``, and a discrimination test flips a
sign to confirm that tolerance can fail.

**No gradient-agreement test here, and that is deliberate.** The topic dossier
scopes this reduced track to the forward pass: minGRU and minLSTM derive no
BPTT, because the hand-derived backward passes of the classical cells already
teach that lesson. The finite-difference check that guards the rest of the
sequence package lives in ``test_lstm.py`` and ``test_gru.py``; what stands in
for it here is the three-way forward agreement plus the reduction tests, which
pin the deletions the paper makes -- no reset gate, no output gate, and above
all **no tanh on the candidate**, the one a later reader is most likely to
"fix" back.

The classical GRU used by the reduction tests is written out locally as an
oracle rather than imported from the sibling cell module, so that this suite
stays runnable on its own, as the dossier's independent track intends.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import numpy as np
import pytest

from dlhub.nn.sequence.min_gated import (
    g,
    identity_candidate,
    log_g,
    log_space_scan,
    logcumsumexp,
    make_fixture,
    min_gru_coefficients,
    min_gru_forward_log,
    min_gru_forward_parallel,
    min_gru_forward_sequential,
    min_gru_log_coefficients,
    min_lstm_coefficients,
    min_lstm_forward_log,
    min_lstm_forward_parallel,
    min_lstm_forward_sequential,
    min_lstm_log_coefficients,
    parallel_scan,
    project_over_time,
    sequential_scan,
    softplus,
)

# The dossier's scan-equivalence contract: the same arithmetic in a different
# association order, so exact equality is not promised and this is.
SCAN_ATOL = 1e-10

# A flipped sign must miss by far more than the tolerance above, or the
# tolerance is not checking anything. Measured discrepancies are ~1e0 here.
DISCRIMINATION_FLOOR = 0.1

CELLS = {
    "minGRU": {
        "weights": "min_gru",
        "coefficients": min_gru_coefficients,
        "log_coefficients": min_gru_log_coefficients,
        "sequential": min_gru_forward_sequential,
        "parallel": min_gru_forward_parallel,
        "log_space": min_gru_forward_log,
    },
    "minLSTM": {
        "weights": "min_lstm",
        "coefficients": min_lstm_coefficients,
        "log_coefficients": min_lstm_log_coefficients,
        "sequential": min_lstm_forward_sequential,
        "parallel": min_lstm_forward_parallel,
        "log_space": min_lstm_forward_log,
    },
}
CELL_NAMES = list(CELLS)


def cell_inputs(cell_name):
    """The fixture arrays one cell needs: ``(x, a0, parameters, cell)``."""
    fixture = make_fixture()
    cell = CELLS[cell_name]
    return fixture["x"], fixture["a0"], fixture[cell["weights"]], cell


def classical_gru_sequence(x, Wr, Wu, Wc, a0):
    """
    The hub's canonical GRU (Cho/Ng form), as an independent oracle.

    Written out here, with the textbook sigmoid and the concatenated weight
    form the topic pins, so the reduction tests below compare the minimal cell
    against the classical equations rather than against itself. The reset gate
    is applied *before* the hidden transform, which is the hub's canonical
    placement.
    """
    n_a, m, T_x = Wu.shape[0], x.shape[1], x.shape[2]
    a = np.zeros((n_a, m, T_x))
    a_prev = a0
    for t in range(T_x):
        stacked = np.concatenate([a_prev, x[:, :, t]], axis=0)
        gamma_r = 1.0 / (1.0 + np.exp(-(Wr @ stacked)))
        gamma_u = 1.0 / (1.0 + np.exp(-(Wu @ stacked)))
        reset_stacked = np.concatenate([gamma_r * a_prev, x[:, :, t]], axis=0)
        a_tilde = np.tanh(Wc @ reset_stacked)
        a_next = gamma_u * a_tilde + (1.0 - gamma_u) * a_prev
        a[:, :, t] = a_next
        a_prev = a_next
    return a


# --- Scan equivalence (the test this module exists to pass) --------------


@pytest.mark.parametrize("cell_name", CELL_NAMES)
def test_the_parallel_scan_matches_the_sequential_recurrence(cell_name):
    """
    The associative scan computes the recurrence, not something adjacent to it.

    Same coefficients, same initial state, different association order. A scan
    that drops the initial state's decay, or composes its pairs the wrong way
    round, lands far outside this tolerance.
    """
    x, a0, parameters, cell = cell_inputs(cell_name)
    reference = cell["sequential"](x, parameters, a0)
    scanned = cell["parallel"](x, parameters, a0)
    np.testing.assert_allclose(scanned, reference, atol=SCAN_ATOL)


@pytest.mark.parametrize("cell_name", CELL_NAMES)
def test_the_parallel_scan_matches_from_the_zero_initial_state(cell_name):
    """The default ``a^{<0>} = 0`` is the case a scan is most likely to skip."""
    x, _, parameters, cell = cell_inputs(cell_name)
    reference = cell["sequential"](x, parameters)
    scanned = cell["parallel"](x, parameters)
    np.testing.assert_allclose(scanned, reference, atol=SCAN_ATOL)


@pytest.mark.parametrize("cell_name", CELL_NAMES)
def test_the_log_space_scan_matches_the_same_cell_in_plain_space(cell_name):
    """
    The stable form is the same cell, not a different one.

    The log-space path needs a positive candidate, so the comparison uses the
    sequential recurrence with that same candidate ``g`` -- comparing it against
    the linear candidate would be comparing two different cells.
    """
    x, a0, parameters, cell = cell_inputs(cell_name)
    positive_a0 = np.abs(a0)  # log space needs a non-negative initial state
    reference = cell["sequential"](x, parameters, positive_a0, candidate=g)
    scanned = cell["log_space"](x, parameters, positive_a0)
    np.testing.assert_allclose(scanned, reference, atol=SCAN_ATOL)


@pytest.mark.parametrize("cell_name", CELL_NAMES)
def test_the_log_coefficients_exponentiate_to_the_plain_ones(cell_name):
    """
    ``log_alpha`` and ``log_beta`` are a rearrangement, not a re-derivation.

    Exponentiating them must return the plain coefficients computed with the
    same candidate, which is what makes the softplus identities in the module
    Notes checkable rather than merely asserted.
    """
    x, _, parameters, cell = cell_inputs(cell_name)
    alpha, beta = cell["coefficients"](x, parameters, candidate=g)
    log_alpha, log_beta = cell["log_coefficients"](x, parameters)
    np.testing.assert_allclose(np.exp(log_alpha), alpha, rtol=1e-12)
    np.testing.assert_allclose(np.exp(log_beta), beta, rtol=1e-12)


@pytest.mark.parametrize("cell_name", CELL_NAMES)
@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 7, 8, 9])
def test_the_scans_hold_at_every_sequence_length(cell_name, length):
    """
    No power-of-two assumption, in either scan.

    The stride-doubling loop is where such an assumption hides: pad-to-8 logic
    passes at length 4 and 8 and fails at 5, 7, and 9. Length 1 is the other
    edge -- the doubling loop never runs at all.
    """
    x, a0, parameters, cell = cell_inputs(cell_name)
    x_short = x[:, :, :length]
    positive_a0 = np.abs(a0)

    plain_reference = cell["sequential"](x_short, parameters, a0)
    np.testing.assert_allclose(
        cell["parallel"](x_short, parameters, a0), plain_reference, atol=SCAN_ATOL
    )

    log_reference = cell["sequential"](x_short, parameters, positive_a0, candidate=g)
    np.testing.assert_allclose(
        cell["log_space"](x_short, parameters, positive_a0),
        log_reference,
        atol=SCAN_ATOL,
    )


def test_the_log_space_scan_survives_a_long_saturated_sequence():
    """
    The regime the log-space form exists for: a decay applied hundreds of times.

    With ``W_u`` strongly negative the update gate is near zero, so ``alpha`` is
    near one and ``beta`` tiny, and the prefix products the scan needs span the
    whole sequence. Both forms must stay finite and stay together.
    """
    rng = np.random.default_rng(0)
    n_x, n_a, m, T_x = 3, 4, 2, 512
    x = rng.standard_normal((n_x, m, T_x))
    parameters = {
        "Wu": np.full((n_a, n_x), -4.0),
        "Wc": rng.standard_normal((n_a, n_x)),
    }
    a_start = np.ones((n_a, m))

    reference = min_gru_forward_sequential(x, parameters, a_start, candidate=g)
    scanned = min_gru_forward_log(x, parameters, a_start)

    assert np.all(np.isfinite(scanned))
    np.testing.assert_allclose(scanned, reference, atol=SCAN_ATOL)


@pytest.mark.parametrize("cell_name", CELL_NAMES)
def test_the_equivalence_check_rejects_a_flipped_sign(cell_name):
    """
    The defect a hub must never ship. Flipping the sign of either coefficient
    is the smallest wrong scan imaginable, and the tolerance above must reject
    it -- a tolerance nothing can fail is not a check.
    """
    x, a0, parameters, cell = cell_inputs(cell_name)
    alpha, beta = cell["coefficients"](x, parameters)
    reference = sequential_scan(alpha, beta, a0)

    flipped_beta = parallel_scan(alpha, -beta, a0)
    flipped_alpha = parallel_scan(-alpha, beta, a0)

    assert np.max(np.abs(reference - flipped_beta)) > DISCRIMINATION_FLOOR
    assert np.max(np.abs(reference - flipped_alpha)) > DISCRIMINATION_FLOOR


# --- Reduction to the classical gates ------------------------------------


def test_min_gru_is_a_classical_gru_whose_gates_ignore_the_previous_state():
    """
    The reduction, stated as an equality against the classical equations.

    Zero the recurrent halves of a classical GRU's update and candidate weights
    -- which is exactly "the gates stop depending on ``a^{<t-1>}``" -- and the
    cell becomes minGRU with a tanh candidate. This is the claim Sec. 3.1 makes,
    checked against an independent implementation of the Cho/Ng form.
    """
    fixture = make_fixture()
    x, a0, dims = fixture["x"], fixture["a0"], fixture["dims"]
    n_a, n_x = dims["n_a"], dims["n_x"]
    rng = np.random.default_rng(0)

    Wux, Wcx = rng.standard_normal((n_a, n_x)), rng.standard_normal((n_a, n_x))
    zeros_half = np.zeros((n_a, n_a))
    classical = classical_gru_sequence(
        x,
        Wr=rng.standard_normal((n_a, n_a + n_x)),  # any reset gate at all
        Wu=np.concatenate([zeros_half, Wux], axis=1),
        Wc=np.concatenate([zeros_half, Wcx], axis=1),
        a0=a0,
    )
    minimal = min_gru_forward_sequential(
        x, {"Wu": Wux, "Wc": Wcx}, a0, candidate=np.tanh
    )
    np.testing.assert_allclose(minimal, classical, atol=1e-12)


def test_the_reset_gate_has_nothing_left_to_reset():
    """
    Why minGRU can drop the reset gate outright (Sec. 3.1.1).

    Once the candidate's recurrent half is zero, ``Gamma_r`` multiplies a state
    that is then multiplied by zero. Two entirely different reset weights must
    give the identical trajectory, which is what makes the gate removable
    rather than merely unused.
    """
    fixture = make_fixture()
    x, a0, dims = fixture["x"], fixture["a0"], fixture["dims"]
    n_a, n_x = dims["n_a"], dims["n_x"]
    rng = np.random.default_rng(1)

    gates = {
        "Wu": np.concatenate(
            [np.zeros((n_a, n_a)), rng.standard_normal((n_a, n_x))], axis=1
        ),
        "Wc": np.concatenate(
            [np.zeros((n_a, n_a)), rng.standard_normal((n_a, n_x))], axis=1
        ),
    }
    first = classical_gru_sequence(
        x, Wr=rng.standard_normal((n_a, n_a + n_x)), a0=a0, **gates
    )
    second = classical_gru_sequence(
        x, Wr=100.0 * rng.standard_normal((n_a, n_a + n_x)), a0=a0, **gates
    )
    np.testing.assert_array_equal(first, second)


def test_the_candidate_is_linear_and_not_tanh():
    """
    Sec. 3.1.2's deletion, pinned so it cannot be helpfully undone.

    ``beta`` must be the update gate times a *plain* ``W_c x``. Restoring the
    classical ``tanh`` would leave every shape intact and every scan
    consistent, so only a test against the linear value catches it -- and the
    second assertion confirms the two really do differ on this fixture.
    """
    fixture = make_fixture()
    x, parameters = fixture["x"], fixture["min_gru"]
    alpha, beta = min_gru_coefficients(x, parameters)
    gamma_u = 1.0 - alpha

    for t in range(fixture["dims"]["T_x"]):
        a_tilde_t = parameters["Wc"] @ x[:, :, t]  # the linear candidate
        np.testing.assert_allclose(beta[:, :, t], gamma_u[:, :, t] * a_tilde_t)

    _, tanh_beta = min_gru_coefficients(x, parameters, candidate=np.tanh)
    assert np.max(np.abs(beta - tanh_beta)) > DISCRIMINATION_FLOOR


def test_min_lstm_gates_are_normalised_to_sum_to_one():
    """
    Sec. 3.2.4's normalisation, which is what stands in for the dropped output
    gate: ``Gamma_f' + Gamma_u' = 1`` exactly, so the recurrence is a convex
    blend and the state keeps a time-independent scale.
    """
    fixture = make_fixture()
    x, parameters = fixture["x"], fixture["min_lstm"]
    alpha, beta = min_lstm_coefficients(x, parameters)

    a_tilde = project_over_time(parameters["Wc"], x)
    gamma_u_prime = beta / a_tilde  # beta = Gamma_u' * a_tilde
    np.testing.assert_allclose(alpha + gamma_u_prime, 1.0, atol=1e-15)


def test_min_lstm_keeps_the_state_inside_its_convex_hull():
    """
    The consequence of that normalisation, and the reason no output gate is
    needed: every state is a weighted average of the initial state and the
    candidates seen so far, so it can never exceed their largest magnitude.
    """
    fixture = make_fixture()
    x, a0, parameters = fixture["x"], fixture["a0"], fixture["min_lstm"]
    a = min_lstm_forward_sequential(x, parameters, a0)
    a_tilde = project_over_time(parameters["Wc"], x)

    bound = max(np.max(np.abs(a0)), np.max(np.abs(a_tilde)))
    assert np.max(np.abs(a)) <= bound + 1e-12


# --- Determinism ---------------------------------------------------------


def test_the_seeded_fixture_reproduces_exactly():
    """The pinned draw order under seed(1) must give bit-identical arrays."""
    first, second = make_fixture(), make_fixture()
    np.testing.assert_array_equal(first["x"], second["x"])
    np.testing.assert_array_equal(first["a0"], second["a0"])
    for cell, keys in (("min_gru", ("Wu", "Wc")), ("min_lstm", ("Wf", "Wu", "Wc"))):
        for key in keys:
            np.testing.assert_array_equal(first[cell][key], second[cell][key])


@pytest.mark.parametrize("cell_name", CELL_NAMES)
def test_the_trajectory_is_deterministic_in_all_three_forms(cell_name):
    """Re-running any of the three forms yields bit-identical states."""
    x, a0, parameters, cell = cell_inputs(cell_name)
    positive_a0 = np.abs(a0)
    for form, state in (
        (cell["sequential"], a0),
        (cell["parallel"], a0),
        (cell["log_space"], positive_a0),
    ):
        np.testing.assert_array_equal(
            form(x, parameters, state), form(x, parameters, state)
        )


# --- Shape and edge contracts --------------------------------------------


def test_the_fixture_has_the_dimensions_the_topic_pins():
    """
    ``T_x = 7`` is load-bearing, not incidental: it is odd and not a power of
    two, which is what makes the length sweep above able to fail.
    """
    fixture = make_fixture()
    dims = fixture["dims"]
    assert (dims["n_x"], dims["n_a"], dims["m"], dims["T_x"]) == (3, 5, 10, 7)
    assert dims["T_x"] % 2 == 1
    assert dims["T_x"] & (dims["T_x"] - 1) != 0  # not a power of two
    assert fixture["x"].shape == (3, 10, 7)
    assert fixture["a0"].shape == (5, 10)
    for key in ("Wu", "Wc"):
        assert fixture["min_gru"][key].shape == (5, 3)
    for key in ("Wf", "Wu", "Wc"):
        assert fixture["min_lstm"][key].shape == (5, 3)


@pytest.mark.parametrize("cell_name", CELL_NAMES)
def test_every_form_produces_the_declared_shape(cell_name):
    """Hidden states are ``(n_a, m, T_x)`` whichever way the scan is taken."""
    x, a0, parameters, cell = cell_inputs(cell_name)
    fixture = make_fixture()
    dims = fixture["dims"]
    expected = (dims["n_a"], dims["m"], dims["T_x"])
    assert cell["sequential"](x, parameters, a0).shape == expected
    assert cell["parallel"](x, parameters, a0).shape == expected
    assert cell["log_space"](x, parameters, np.abs(a0)).shape == expected


@pytest.mark.parametrize("cell_name", CELL_NAMES)
def test_the_coefficients_carry_the_state_shape(cell_name):
    """``alpha`` and ``beta`` are per-unit, per-example, per-step."""
    x, _, parameters, cell = cell_inputs(cell_name)
    dims = make_fixture()["dims"]
    expected = (dims["n_a"], dims["m"], dims["T_x"])
    for coefficient in cell["coefficients"](x, parameters):
        assert coefficient.shape == expected
    for coefficient in cell["log_coefficients"](x, parameters):
        assert coefficient.shape == expected


def test_project_over_time_matches_a_per_timestep_matmul():
    """
    The one contraction the whole reduction rests on: applying ``W`` to every
    timestep at once must equal applying it to each in turn.
    """
    fixture = make_fixture()
    x, W = fixture["x"], fixture["min_gru"]["Wu"]
    projected = project_over_time(W, x)
    for t in range(fixture["dims"]["T_x"]):
        np.testing.assert_allclose(projected[:, :, t], W @ x[:, :, t])


@pytest.mark.parametrize("cell_name", CELL_NAMES)
def test_the_default_initial_state_is_the_zero_vector(cell_name):
    """Omitting ``a0`` matches passing an explicit zero state."""
    x, _, parameters, cell = cell_inputs(cell_name)
    dims = make_fixture()["dims"]
    zeros = np.zeros((dims["n_a"], dims["m"]))
    np.testing.assert_array_equal(
        cell["sequential"](x, parameters), cell["sequential"](x, parameters, zeros)
    )
    np.testing.assert_array_equal(
        cell["log_space"](x, parameters), cell["log_space"](x, parameters, zeros)
    )


def test_a_single_timestep_sequence_works():
    """
    ``T_x = 1`` is the degenerate edge: the doubling loop never runs, so the
    scan must already hold the single step it started with.
    """
    fixture = make_fixture()
    x1, a0, parameters = fixture["x"][:, :, :1], fixture["a0"], fixture["min_gru"]
    dims = fixture["dims"]

    sequential = min_gru_forward_sequential(x1, parameters, a0)
    assert sequential.shape == (dims["n_a"], dims["m"], 1)
    np.testing.assert_allclose(
        min_gru_forward_parallel(x1, parameters, a0), sequential, atol=SCAN_ATOL
    )


def test_a_forward_pass_rejects_a_two_dimensional_input():
    fixture = make_fixture()
    with pytest.raises(ValueError):
        min_gru_forward_sequential(fixture["x"][:, :, 0], fixture["min_gru"])


@pytest.mark.parametrize("scan", [sequential_scan, parallel_scan])
def test_a_scan_rejects_a_mismatched_initial_state(scan):
    alpha = np.ones((5, 10, 7))
    with pytest.raises(ValueError):
        scan(alpha, alpha, np.zeros((99, 10)))


@pytest.mark.parametrize("scan", [sequential_scan, parallel_scan, log_space_scan])
def test_a_scan_rejects_mismatched_coefficients(scan):
    with pytest.raises(ValueError):
        scan(np.ones((5, 10, 7)), np.ones((5, 10, 6)))


def test_the_log_space_scan_rejects_a_negative_initial_state():
    """
    The contract the log-space form cannot bend: ``log`` of a negative state
    does not exist, so a negative ``a0`` is a caller error rather than a NaN
    handed back three functions later.
    """
    fixture = make_fixture()
    with pytest.raises(ValueError):
        min_gru_forward_log(fixture["x"], fixture["min_gru"], fixture["a0"])


@pytest.mark.parametrize("cell_name", CELL_NAMES)
def test_the_log_space_scan_returns_a_strictly_positive_state(cell_name):
    """
    With the positive candidate ``g`` and a non-negative start, every state
    stays positive -- which is the invariant that lets the next step take its
    logarithm.
    """
    x, a0, parameters, cell = cell_inputs(cell_name)
    a = cell["log_space"](x, parameters, np.abs(a0))
    assert np.all(a > 0)


# --- Elementwise primitives ----------------------------------------------


def test_softplus_is_stable_where_the_textbook_form_overflows():
    """
    ``log(1 + exp(z))`` overflows around ``z = 710``; the shifted form does not,
    and for large positive ``z`` it must return ``z`` itself.
    """
    moderate = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    np.testing.assert_allclose(softplus(moderate), np.log1p(np.exp(moderate)))

    extreme = np.array([-800.0, 800.0])
    values = softplus(extreme)
    assert np.all(np.isfinite(values))
    np.testing.assert_allclose(values, [0.0, 800.0], atol=1e-12)


def test_the_gate_log_identities_hold():
    """
    The two identities the log-space coefficients are built from:
    ``log Gamma_u = -softplus(-k)`` and ``log(1 - Gamma_u) = -softplus(k)``.

    The second half of the test is the reason the identities are used at all.
    At ``k = 50`` the textbook complement ``1 - sigma(k)`` rounds to exactly
    zero and its logarithm is ``-inf``, while ``-softplus(k)`` still carries
    the true value ``e^{-50}``: a closed gate is small, not absent.
    """
    k = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    gamma = 1.0 / (1.0 + np.exp(-k))
    np.testing.assert_allclose(np.exp(-softplus(-k)), gamma, rtol=1e-12)
    np.testing.assert_allclose(np.exp(-softplus(k)), 1.0 - gamma, rtol=1e-12)

    saturated = np.array([50.0])
    assert 1.0 - 1.0 / (1.0 + np.exp(-saturated)) == 0.0
    np.testing.assert_allclose(
        np.exp(-softplus(saturated)), np.exp(-saturated), rtol=1e-12
    )


def test_g_is_positive_continuous_and_linear_above_zero():
    """
    Sec. B.2.1's candidate: strictly positive everywhere, ``k + 0.5`` on the
    non-negative half, and continuous where the two branches meet.
    """
    k = np.linspace(-6.0, 6.0, 25)
    values = g(k)
    assert np.all(values > 0)
    non_negative = k >= 0
    np.testing.assert_allclose(values[non_negative], k[non_negative] + 0.5)
    assert g(np.array([0.0]))[0] == pytest.approx(0.5)
    assert g(np.array([-1e-12]))[0] == pytest.approx(0.5, abs=1e-9)


def test_log_g_is_the_log_of_g():
    """Both branches of ``log_g`` must agree with taking the log the slow way."""
    k = np.linspace(-20.0, 20.0, 41)
    np.testing.assert_allclose(np.exp(log_g(k)), g(k), rtol=1e-12)
    np.testing.assert_allclose(log_g(k), np.log(g(k)), rtol=1e-12)


def test_the_identity_candidate_is_the_identity():
    """The deletion has a name; it must also have no effect."""
    k = np.linspace(-3.0, 3.0, 7)
    np.testing.assert_array_equal(identity_candidate(k), k)


def test_logcumsumexp_matches_the_naive_form():
    """
    Checked against ``log(cumsum(exp(z)))`` in the range where that form is
    safe, then against a shifted copy where it is not: adding a constant to
    every entry must shift the result by exactly that constant.
    """
    rng = np.random.default_rng(2)
    z = rng.standard_normal((3, 4, 7))
    np.testing.assert_allclose(
        logcumsumexp(z), np.log(np.cumsum(np.exp(z), axis=-1)), rtol=1e-12
    )
    np.testing.assert_allclose(logcumsumexp(z + 900.0), logcumsumexp(z) + 900.0)


def test_logcumsumexp_treats_negative_infinity_as_a_zero_term():
    """
    ``-inf`` is how a zero initial state enters the log-space scan, so it has to
    contribute nothing rather than poison the running total with a NaN.
    """
    z = np.array([[-np.inf, 0.0, 1.0]])
    result = logcumsumexp(z)
    assert result[0, 0] == -np.inf
    expected = np.log(np.cumsum(np.exp([0.0, 1.0])))  # the -inf term drops out
    np.testing.assert_allclose(result[0, 1:], expected)
