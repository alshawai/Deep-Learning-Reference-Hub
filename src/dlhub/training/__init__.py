"""
Training Techniques
===================

Techniques that wrap a training run rather than perform it: stopping it at the
right moment, and verifying that the gradients driving it are correct.

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

from dlhub.training.gradient_checking import (
    dictionary_to_vector,
    gradient_check,
    vector_to_dictionary,
)

__all__ = [
    "dictionary_to_vector",
    "gradient_check",
    "vector_to_dictionary",
]
