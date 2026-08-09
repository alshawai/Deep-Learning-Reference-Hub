"""
Early Stopping Utility Tests
=============================

`early_stopping` is a pure function over a validation-loss history, so every
case is a closed-form expectation on a hand-built list. The one behaviour worth
guarding closely is what counts as "best": the docstring promises `min_delta`
guards against noise, and it is easy to write that check backwards.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

from conftest import load

early_stopping = load("early_stopping.py").early_stopping


def test_fewer_than_two_epochs_never_stops():
    should_stop, info = early_stopping([1.0], patience=1, verbose=False)
    assert should_stop is False
    assert "message" in info


def test_stops_once_patience_epochs_pass_without_improvement():
    losses = [1.0, 0.9, 0.9, 0.9, 0.9]  # best at epoch 1, flat for 3 epochs since
    should_stop, info = early_stopping(losses, patience=3, verbose=False)
    assert should_stop is True
    assert info["epochs_since_improvement"] == 3
    assert info["best_epoch"] == 1


def test_does_not_stop_one_epoch_before_patience_is_reached():
    losses = [1.0, 0.9, 0.9, 0.9]
    should_stop, info = early_stopping(losses, patience=3, verbose=False)
    assert should_stop is False
    assert info["patience_remaining"] == 1


def test_a_still_improving_run_never_stops():
    losses = [1.0, 0.5, 0.4, 0.3, 0.2, 0.1]
    should_stop, info = early_stopping(losses, patience=3, verbose=False)
    assert should_stop is False
    assert info["epochs_since_improvement"] == 0
    assert info["best_epoch"] == len(losses) - 1


def test_best_loss_is_the_best_seen_not_the_most_recent():
    losses = [0.5, 0.2, 0.4, 0.4]
    _, info = early_stopping(losses, patience=10, verbose=False)
    assert info["best_loss"] == 0.2
    assert info["best_epoch"] == 1
    assert info["current_loss"] == 0.4


def test_best_loss_and_best_epoch_always_refer_to_the_same_entry():
    """
    The two are reported together, so they have to describe one epoch. Computing
    them from different scans is how they drift apart.
    """
    for losses in (
        [0.5, 0.2, 0.4, 0.4],
        [1.0, 0.9, 0.9, 0.5, 0.5],
        [1.0, 0.9999, 0.9998],
        [0.3, 0.7, 0.9],
    ):
        for min_delta in (0.0, 1e-4, 0.01):
            _, info = early_stopping(
                losses, patience=99, min_delta=min_delta, verbose=False
            )
            assert losses[info["best_epoch"]] == info["best_loss"], (
                f"{losses} at min_delta={min_delta}"
            )


def test_a_new_minimum_resets_the_improvement_counter():
    losses = [1.0, 0.9, 0.9, 0.5, 0.5]
    _, info = early_stopping(losses, patience=3, verbose=False)
    assert info["epochs_since_improvement"] == 1
    assert info["best_epoch"] == 3


def test_a_smaller_min_delta_does_not_move_the_best_loss():
    """
    When every step clears both bars the two agree, so `min_delta` is invisible
    here. The cases where it is not are the two tests below.
    """
    losses = [1.0, 0.9, 0.8]
    _, tight = early_stopping(losses, patience=10, min_delta=1e-6, verbose=False)
    _, loose = early_stopping(losses, patience=10, min_delta=0.05, verbose=False)
    assert tight["best_loss"] == loose["best_loss"] == 0.8


def test_a_drift_smaller_than_min_delta_does_not_reset_patience():
    """
    Regression test. `min_delta` is documented as the minimum change required for
    improvement, and the whole reason to have it is that validation loss wobbles.
    Taking the best as a plain minimum ignores it: a run that inches down by
    1e-12 an epoch sets a new record every epoch, `epochs_since_improvement`
    never leaves 0, and training that has plateaued runs forever.
    """
    losses = [1.0 - 1e-12 * i for i in range(30)]
    should_stop, info = early_stopping(losses, patience=5, verbose=False)
    assert info["epochs_since_improvement"] == 29
    assert should_stop is True


def test_an_improvement_larger_than_min_delta_does_reset_patience():
    """The other half of the same rule: real progress must still count."""
    losses = [1.0, 0.99, 0.98, 0.5, 0.5]
    should_stop, info = early_stopping(
        losses, patience=3, min_delta=0.05, verbose=False
    )
    assert info["best_epoch"] == 3
    assert info["epochs_since_improvement"] == 1
    assert should_stop is False


def test_min_delta_zero_accepts_any_decrease():
    losses = [1.0, 1.0 - 1e-12]
    _, info = early_stopping(losses, patience=5, min_delta=0.0, verbose=False)
    assert info["best_epoch"] == 1
    assert info["epochs_since_improvement"] == 0


def test_improvement_needed_is_best_loss_minus_min_delta():
    _, info = early_stopping([1.0, 0.5], patience=5, min_delta=0.1, verbose=False)
    assert info["improvement_needed"] == 0.5 - 0.1


def test_patience_remaining_never_goes_negative():
    losses = [1.0] + [1.0] * 10  # 10 epochs with no improvement, patience 3
    _, info = early_stopping(losses, patience=3, verbose=False)
    assert info["patience_remaining"] == 0


def test_verbose_false_prints_nothing(capsys):
    early_stopping([1.0, 0.9, 0.9], patience=1, verbose=False)
    assert capsys.readouterr().out == ""


def test_verbose_true_prints_a_report(capsys):
    early_stopping([1.0, 0.9, 0.9], patience=1, verbose=True)
    assert "Early Stopping Check" in capsys.readouterr().out
