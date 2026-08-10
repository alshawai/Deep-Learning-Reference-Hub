"""
Early Stopping Utility
======================

Implements a mechanism to halt model training when a monitored validation metric
(such as loss or accuracy) ceases to improve after a specified number of epochs.

This prevents overfitting and saves compute by terminating training once performance plateaus.

References
----------
- Prechelt, L. (2012). Early Stopping — But When? In *Neural Networks: Tricks of the Trade*. Springer.
  https://link.springer.com/chapter/10.1007/978-3-642-35289-8_5

Author
------
Deep Learning Reference Hub

License
-------
MIT License

Notes
-----
- Works by monitoring "no improvement" for `patience` consecutive epochs.
- An epoch improves only if it beats the running best by more than `min_delta`.
  So `best_loss` is the best value that cleared that bar, which is not always the
  smallest value in the history: under a large `min_delta`, a slow drift downward
  never clears it and the recorded best stays where it was. This is the same rule
  Keras applies, and it is what makes `min_delta` a noise filter rather than a
  number that only appears in the report.
"""


def early_stopping(
    val_losses: list[float],
    patience: int = 10,
    min_delta: float = 1e-4,
    verbose: bool = True,
) -> tuple[bool, dict]:
    """
    Early stopping with detailed tracking and optional verbose output.

    Parameters
    ----------
    val_losses : list
        Validation losses from training history
    patience : int, default=10
        Number of epochs to wait after last improvement
    min_delta : float, default=1e-4
        Minimum decrease that counts as an improvement. A loss that falls by less
        than this is treated as noise, so it does not reset the patience counter.
    verbose : bool, default=True
        Whether to print detailed information

    Returns
    -------
        tuple[bool, dict]
            whether to stop and info dict, which contains:
            - 'best_loss': Best validation loss seen so far
            - 'best_epoch': Epoch that produced it
            - 'epochs_since_improvement': Number of epochs since last improvement
            - 'current_loss': Most recent validation loss
            - 'improvement_needed': Loss the next epoch must beat to count
            - 'patience_remaining': Epochs left before stopping
    """
    if len(val_losses) < 2:
        return False, {"message": "Need at least 2 epochs to evaluate"}

    # Scanned rather than taken as min(val_losses), because `min_delta` decides
    # what counts as an improvement. A run that drifts down by less than
    # min_delta each epoch has a new minimum every epoch and would never stop,
    # which is the noise this parameter exists to reject.
    best_loss = val_losses[0]
    best_epoch = 0
    for epoch, loss in enumerate(val_losses[1:], start=1):
        if loss < best_loss - min_delta:
            best_loss = loss
            best_epoch = epoch

    current_epoch = len(val_losses) - 1
    epochs_since_improvement = current_epoch - best_epoch
    current_loss = val_losses[-1]
    improvement_needed = best_loss - min_delta

    info = {
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "epochs_since_improvement": epochs_since_improvement,
        "current_loss": current_loss,
        "improvement_needed": improvement_needed,
        "patience_remaining": max(0, patience - epochs_since_improvement),
    }

    should_stop = epochs_since_improvement >= patience

    if verbose:
        print("Early Stopping Check:")
        print(f"  Current Loss: {current_loss:.6f}")
        print(f"  Best Loss: {best_loss:.6f} (epoch {best_epoch})")
        print(f"  Epochs since improvement: {epochs_since_improvement}")
        print(f"  Patience remaining: {info['patience_remaining']}")
        print(f"  Should stop: {should_stop}")

    return should_stop, info
