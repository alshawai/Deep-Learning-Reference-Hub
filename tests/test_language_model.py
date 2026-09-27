"""
Character-Level Language Model -- Own-Correctness Tests
=======================================================

The language-modeling topic has **no from-scratch NumPy reference**: it adds no
new differentiable mathematics over the vanilla RNN cell, so PyTorch is its
canonical -- and only -- implementation. There is therefore nothing to hold a
*parity* check against. This suite instead pins the module's **own correctness**,
the anchors the dossier fixes: the ``ln V`` initialisation anchor, a strictly
decreasing training loss, byte-for-byte reproducible sampling, a shuffled-target
discrimination control, and a cross-check of ``clip_grad_norm_`` against the
``recurrent-neural-networks`` topic's from-scratch ``clip_gradients``. Each check
is designed to *fail* on a real mistake -- "it runs" is not one of them.

This is an idiom track, not a parity port, so the tests do not match gradients
against the framework. The one genuine numeric cross-check is the clip: it is a
single primitive, not a whole-model gradient, and it turns up a convention a
practitioner needs -- ``clip_grad_norm_`` divides by ``||g|| + 1e-6``, so on
ordinary-scale gradients it agrees with ``clip_gradients`` only to ``~1e-8``, not
the float64 ``1e-10`` the dossier names; the ``1e-10`` rule holds once the
gradient norm dwarfs that fixed epsilon.

The tests live in a class whose name matches ``-k parity`` so the cross-framework
CI job -- the only job that installs ``torch`` -- selects and runs them. Named
any other way, the file would collect in the framework-free job, skip at the
``importorskip`` below, and read green without ever executing. A skip is not a
pass here.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import numpy as np
import pytest

# Skip the whole module when torch is absent so the framework-free job can
# collect it. The cross-framework CI job installs torch as an explicit step and
# selects this file by its class name under ``-k parity``, so there the import
# succeeds and every assertion below runs for real.
torch = pytest.importorskip("torch")

from dlhub.nn.sequence.rnn import clip_gradients  # noqa: E402  (after the gate)
from dlhub.pytorch.sequence.language_model import (  # noqa: E402  (after the gate)
    build_model,
    corpus_cross_entropy,
    make_fixture,
    perplexity,
    sample,
    train,
)

# --- Declared tolerances and training regime -----------------------------

# The initialisation anchor: the deviation of the init cross-entropy from ln V is
# second-order in the (small) default-init logits, so any correct read-out lands
# within ~0.1 nats; a summed-not-mean reduction (~12 nats here) or a log base
# other than e misses by nats, far outside this band.
INIT_ANCHOR_ATOL = 0.1

# The training-smoke regime. The module's default learning rate (0.1) makes
# full-corpus descent on this fixture strictly monotonic; 150 steps is enough for
# a loss drop no numerical noise could explain.
TRAIN_STEPS = 150
MIN_LOSS_DROP = 0.5

# The clip cross-check. The global-norm *rule* matches clip_gradients to float64
# precision only where torch's fixed 1e-6 denominator epsilon is negligible
# (large gradient norm); on ordinary-scale gradients the epsilon is the whole
# discrepancy, landing between the two bounds below.
CLIP_RULE_TOLERANCE = 1e-10
CLIP_EPSILON_FLOOR = 1e-10
CLIP_GROSS_CEILING = 1e-6


# --- Shared state, built once --------------------------------------------


@pytest.fixture(scope="module")
def fixture():
    """The deterministic corpus fixture (built once for the module)."""
    return make_fixture()


@pytest.fixture(scope="module")
def trained(fixture):
    """
    A model trained once on the fixture, with the pre-training anchors captured.

    The initial cross-entropy is read *before* ``train`` mutates the weights, so
    the init-anchor test sees the untrained read-out; the loss history and the
    trained model back the training-smoke, discrimination, and sampling checks.
    """
    model = build_model(fixture["vocab_size"], fixture["hidden_size"])
    with torch.no_grad():
        initial_ce = float(corpus_cross_entropy(model, fixture["examples"]))
    history = train(model, fixture["examples"], steps=TRAIN_STEPS)
    return {"model": model, "initial_ce": initial_ce, "history": history}


def _global_norm(arrays):
    """L2 norm of a dict of arrays viewed as one concatenated vector."""
    return float(np.sqrt(sum(np.sum(a**2) for a in arrays.values())))


def _clip_with_torch(gradients, max_norm):
    """
    Clip a gradient dict with ``torch.nn.utils.clip_grad_norm_``, in float64.

    Loads each array as the ``.grad`` of a leaf tensor -- exactly what the utility
    expects to find on a parameter -- clips in place, and reads the results back
    as NumPy so they compare directly against the from-scratch ``clip_gradients``.
    """
    tensors = []
    for array in gradients.values():
        leaf = torch.zeros(array.shape, dtype=torch.float64, requires_grad=True)
        leaf.grad = torch.tensor(array, dtype=torch.float64)
        tensors.append(leaf)
    torch.nn.utils.clip_grad_norm_(tensors, max_norm)
    return {key: tensors[i].grad.numpy() for i, key in enumerate(gradients)}


def _relative_error(left, right):
    """Norm-based relative error over two matching gradient dicts, bounded by 1."""
    flat_left = np.concatenate([left[key].ravel() for key in left])
    flat_right = np.concatenate([right[key].ravel() for key in left])
    numerator = np.linalg.norm(flat_left - flat_right)
    denominator = np.linalg.norm(flat_left) + np.linalg.norm(flat_right)
    return 0.0 if denominator == 0 else float(numerator / denominator)


def _synthetic_gradients(scale):
    """
    A deterministic gradient dict in the RNN family's parameter shapes.

    Seeded so both clip implementations see the identical arrays; ``scale`` sets
    the global norm, which is the knob that decides whether torch's 1e-6 epsilon
    matters relative to it.
    """
    rng = np.random.default_rng(0)
    shapes = {
        "dWax": (32, 18),
        "dWaa": (32, 32),
        "dWya": (18, 32),
        "dba": (32, 1),
        "dby": (18, 1),
    }
    return {key: rng.standard_normal(shape) * scale for key, shape in shapes.items()}


class TestLanguageModelParity:
    """
    The canonical char-LM's own-correctness checks.

    Grouped under a ``parity``-matching class name so the cross-framework CI job
    (``pytest -k parity``) -- the only one with torch installed -- selects the
    whole file. These are idiom-track correctness anchors, not a gradient-parity
    port; see the module docstring.
    """

    # --- The initialisation anchor ---------------------------------------

    def test_init_cross_entropy_is_finite_and_near_ln_v(self, fixture, trained):
        """
        An untrained char-LM is ~uniform over V, so its per-character
        cross-entropy is finite and ~ ln V (perplexity ~ V). This stands in for
        the parity check a NumPy reference would give: a wrong vocabulary size, a
        loss not mean-reduced over tokens, or a non-natural log misses ln V by
        nats.
        """
        initial_ce = trained["initial_ce"]
        ln_v = np.log(fixture["vocab_size"])

        assert np.isfinite(initial_ce)
        assert initial_ce == pytest.approx(ln_v, abs=INIT_ANCHOR_ATOL)
        assert perplexity(initial_ce) == pytest.approx(
            fixture["vocab_size"], abs=fixture["vocab_size"] * 0.05
        )

    # --- Training smoke ---------------------------------------------------

    def test_training_loss_strictly_decreases(self, trained):
        """
        Full-corpus descent with clipping drives the loss down every step.

        Strict monotonicity, not a curve match: a broken backward pass, a wrong
        loss sign, or clipping that zeroed the update would flatten or raise the
        curve. The total drop must clear a floor no float noise could produce.
        """
        history = trained["history"]
        steps = [history[i + 1] - history[i] for i in range(len(history) - 1)]

        assert all(delta < 0 for delta in steps), (
            f"largest step delta {max(steps):+.2e}"
        )
        assert history[0] - history[-1] > MIN_LOSS_DROP

    # --- Discrimination ---------------------------------------------------

    def test_loss_rejects_a_shuffled_target_control(self, fixture, trained):
        """
        The trained loss must score a mislabelled control worse than the truth.

        Permuting each word's targets breaks the input->next-char correspondence
        the model learned, so its per-character cross-entropy jumps. A metric that
        could not tell the two apart would pass anything.
        """
        model, examples = trained["model"], fixture["examples"]
        generator = torch.Generator().manual_seed(0)
        shuffled = [
            (inputs, targets[torch.randperm(len(targets), generator=generator)])
            for inputs, targets in examples
        ]

        with torch.no_grad():
            correct_ce = float(corpus_cross_entropy(model, examples))
            shuffled_ce = float(corpus_cross_entropy(model, shuffled))

        assert shuffled_ce > correct_ce + MIN_LOSS_DROP
        assert perplexity(shuffled_ce) > perplexity(correct_ce)

    # --- Sampling reproducibility ----------------------------------------

    @pytest.mark.parametrize("temperature", [1.0, 0.5])
    def test_sampling_is_byte_for_byte_reproducible(
        self, fixture, trained, temperature
    ):
        r"""
        A seeded generator makes a sample exactly repeatable.

        Re-seeding the explicitly-passed generator and drawing again must yield an
        identical string; if the RNG were global or implicit, the second draw
        would diverge. The sample also terminates -- on ``\n`` or the cap.
        """
        model = trained["model"]
        max_length = 50

        first = sample(
            model,
            fixture["char_to_ix"],
            fixture["ix_to_char"],
            generator=torch.Generator().manual_seed(fixture["sample_seed"]),
            temperature=temperature,
            max_length=max_length,
        )
        second = sample(
            model,
            fixture["char_to_ix"],
            fixture["ix_to_char"],
            generator=torch.Generator().manual_seed(fixture["sample_seed"]),
            temperature=temperature,
            max_length=max_length,
        )

        assert first == second
        assert 0 < len(first) <= max_length
        assert first.endswith("\n") or len(first) == max_length

    def test_non_positive_temperature_is_rejected(self, fixture, trained):
        """
        ``z / T`` divides by zero at ``T = 0`` and flips the distribution for
        ``T < 0``; the sampler refuses both rather than return nonsense.
        """
        with pytest.raises(ValueError, match="temperature"):
            sample(
                trained["model"],
                fixture["char_to_ix"],
                fixture["ix_to_char"],
                generator=torch.Generator().manual_seed(fixture["sample_seed"]),
                temperature=0.0,
            )

    # --- The clip cross-check (the one genuine numeric agreement) ---------

    def test_clip_matches_numpy_in_the_epsilon_negligible_regime(self):
        """
        Where torch's 1e-6 denominator epsilon is negligible against the gradient
        norm, ``clip_grad_norm_`` reproduces the from-scratch global-norm rule to
        the dossier's float64 ``1e-10``. This is the evidence the two implement the
        *same* operation.
        """
        gradients = _synthetic_gradients(scale=1e4)
        max_norm = _global_norm(gradients) / 3.0  # below the norm, so clipping fires

        clipped_numpy = clip_gradients(gradients, max_norm)
        clipped_torch = _clip_with_torch(gradients, max_norm)

        assert _relative_error(clipped_numpy, clipped_torch) < CLIP_RULE_TOLERANCE

    def test_clip_grad_norm_carries_a_safety_epsilon(self):
        """
        On ordinary-scale gradients the agreement is only ~1e-8, not 1e-10.

        ``clip_grad_norm_`` scales by ``max_norm / (||g|| + 1e-6)`` where
        ``clip_gradients`` scales by ``max_norm / ||g||``. The gap sits above the
        float64 floor and far below any gross error -- the framework convention a
        practitioner reaching for the utility should know about.
        """
        gradients = _synthetic_gradients(scale=1.0)
        max_norm = _global_norm(gradients) / 3.0

        clipped_numpy = clip_gradients(gradients, max_norm)
        clipped_torch = _clip_with_torch(gradients, max_norm)
        error = _relative_error(clipped_numpy, clipped_torch)

        assert CLIP_EPSILON_FLOOR < error < CLIP_GROSS_CEILING

    def test_clip_actually_rescales_an_exceeding_gradient(self):
        """
        Guards the cross-check from passing vacuously: the gradient norm exceeds
        ``max_norm`` before clipping and equals it after, and the clipped values
        differ from the originals by order one -- so an implementation that
        skipped the rescale would be caught, not silently agreed with.
        """
        gradients = _synthetic_gradients(scale=1.0)
        before = _global_norm(gradients)
        max_norm = before / 3.0

        clipped_numpy = clip_gradients(gradients, max_norm)

        assert before > max_norm
        assert _global_norm(clipped_numpy) == pytest.approx(max_norm, rel=1e-9)
        assert _relative_error(gradients, clipped_numpy) > 0.1
