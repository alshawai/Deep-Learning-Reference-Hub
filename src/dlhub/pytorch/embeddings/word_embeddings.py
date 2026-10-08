"""
Word Embeddings -- PyTorch idiom port
=====================================

The static word-embedding reference of :mod:`dlhub.embeddings.word_embeddings`,
rewritten the way it is really written in PyTorch. The NumPy module spells the
lookup identity out as a matrix product ``e_w = E o_w`` to make the mathematics
visible; this port shows the shape of it a practitioner actually ships: a
:class:`torch.nn.Embedding` whose weight *is* the reference matrix (transposed to
PyTorch's row convention), and a gather by index where the NumPy module writes
the one-hot product.

This is an **idiom track**, not a parity port. A lookup is a gather -- there is
no nontrivial computation to agree on -- so the port is not held to a numeric
tolerance against the reference. What it owes instead is the *convention* a reader
must get right and the ``nn.Embedding`` arguments that bite.

The convention is a transpose. Ng's Course-5 embedding matrix ``E`` has shape
``(d, V)`` -- ``d`` features by ``V`` words, one word per **column**, so
``e_w = E[:, i]``. ``torch.nn.Embedding`` stores its weight the other way up, as
``(num_embeddings, embedding_dim) = (V, d)`` -- one word per **row** -- so a word
vector is a row ``weight[i, :]`` and ``weight`` equals ``E.T``. Loading the
reference matrix therefore transposes it; :func:`embedding_from_reference` does
this once, in one place, so a reader who conflates the two conventions does not
silently transpose their own matrix.

References
----------
- Ng, A. (2018). Deep Learning Specialization, Course 5 (Sequence Models),
  Week 2 -- Natural Language Processing & Word Embeddings.
  https://www.coursera.org/specializations/deep-learning
- PyTorch. torch.nn.Embedding.
  https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html

Author
------
Deep Learning Reference Hub

License
-------
MIT

Notes
-----
- **The ``(V, d)`` transpose.** ``weight == E.T``. This is the one fact the port
  exists to teach; every function below works in the ``(V, d)`` row convention,
  and the divergence from Ng's ``(d, V)`` is called out where it would otherwise
  bite.
- **``from_pretrained`` freezes by default.** ``nn.Embedding.from_pretrained``
  sets ``freeze=True`` (``requires_grad=False``) -- the *opposite* of the
  ``nn.Embedding(V, d)`` constructor, whose random weight trains. Loading
  pretrained vectors meaning to fine-tune them is the common place this bites:
  pass ``freeze=False``. :func:`embedding_from_reference` surfaces the argument
  with the framework's own default.
- **dtype is preserved, deliberately.** The reference fixture is ``float64`` and
  this port keeps it so, which is why a :func:`lookup` equals the reference's
  ``lookup`` *exactly* rather than to a tolerance -- a gather copies values and
  does no arithmetic, so there is no rounding to absorb. A real model holds the
  table in ``float32``; the exactness here is a property of the fixture, not a
  claim about production embeddings.
- **Weight tying (forward pointer).** Because ``weight`` is ``(V, d)``, it is also
  exactly the shape of the output projection in a language model: an
  ``nn.Linear(d, V)`` carries a ``(V, d)`` weight too. Sharing one tensor between
  the input embedding and the output softmax is *weight tying*; it drops a
  language model's largest parameter block and couples "the vector that *means*
  word ``v``" to "the vector that *scores* word ``v``". :func:`tied_readout`
  builds that shared read-out; the model it feeds is the sibling
  ``language-modeling-and-sampling`` topic.
"""

import numpy as np
import torch
from torch import nn


def embedding_from_reference(
    E: np.ndarray,
    *,
    freeze: bool = True,
    padding_idx: int | None = None,
) -> nn.Embedding:
    """
    Load Ng's ``(d, V)`` embedding matrix into a ``torch.nn.Embedding``.

    Transposes ``E`` to PyTorch's ``(V, d)`` row convention and hands it to
    :meth:`torch.nn.Embedding.from_pretrained`, so the resulting layer's
    ``weight`` equals ``E.T`` and ``weight[i, :]`` is word ``i``'s vector. The
    input dtype is preserved (``float64`` for the hub fixture), which is what
    makes a subsequent :func:`lookup` match the reference's ``lookup`` exactly.

    Parameters
    ----------
    E : np.ndarray
        The reference embedding matrix, shape ``(d, V)`` -- one word per column,
        Ng's convention.
    freeze : bool, optional
        Whether the returned layer's weight is held constant
        (``requires_grad=False``). Defaults to ``True`` -- ``from_pretrained``'s
        own default, the opposite of the ``nn.Embedding(V, d)`` constructor. Pass
        ``False`` to fine-tune the loaded vectors.
    padding_idx : int or None, optional
        If given, the row whose gradient is held at zero -- the embedding of a
        padding token, which must not drift during training. Defaults to ``None``.

    Returns
    -------
    torch.nn.Embedding
        An embedding layer of ``num_embeddings=V`` rows by ``embedding_dim=d``
        columns, with ``weight == E.T``.

    Raises
    ------
    ValueError
        If ``E`` is not a 2-D array.
    """
    if E.ndim != 2:
        raise ValueError(f"E must be 2-D (d, V); got ndim {E.ndim}")

    # Ng stores a word per column (d, V); torch stores a word per row (V, d). The
    # transpose is the whole lesson -- weight == E.T. ``torch.tensor`` copies, so
    # the layer does not alias the caller's array.
    weight = torch.tensor(np.asarray(E).T)
    return nn.Embedding.from_pretrained(weight, freeze=freeze, padding_idx=padding_idx)


def lookup(layer: nn.Embedding, index: int) -> torch.Tensor:
    """
    Gather one word's embedding by index -- the torch idiom for ``e_w = E o_w``.

    Returns ``weight[index, :]`` by calling the layer on a one-element index
    batch, which is what every embedding layer does in place of materializing the
    one-hot product: a gather, not a matrix multiply. The result does not track
    gradients, so it is a plain copy of the stored vector.

    Parameters
    ----------
    layer : torch.nn.Embedding
        An embedding layer, e.g. from :func:`embedding_from_reference`.
    index : int
        The word's row index ``i`` in ``[0, V)``.

    Returns
    -------
    torch.Tensor
        The embedding ``weight[index, :]``, shape ``(d,)`` -- equal to the
        reference's ``E[:, index]``.
    """
    with torch.no_grad():
        return layer(torch.tensor([index]))[0]


def tied_readout(layer: nn.Embedding, *, bias: bool = False) -> nn.Linear:
    """
    Build an output projection that *shares* the embedding's weight (weight tying).

    The embedding stores ``weight`` of shape ``(V, d)``; an ``nn.Linear(d, V)``
    carries a weight of the same shape, so the two can be one tensor. The returned
    linear layer reuses ``layer.weight`` by reference, so its logits are
    ``h @ weight.T`` -- the score it gives word ``v`` is ``h . e_v``, the dot
    product of the hidden state with word ``v``'s own embedding. This is the
    *weight tying* of a language model's input and output embeddings; see the
    sibling ``language-modeling-and-sampling`` topic for the model it sits in.

    Parameters
    ----------
    layer : torch.nn.Embedding
        The embedding whose ``(V, d)`` weight the read-out shares.
    bias : bool, optional
        Whether the read-out carries its own bias term. Defaults to ``False`` --
        the tied weight is the shared part; a bias, if any, is not tied.

    Returns
    -------
    torch.nn.Linear
        A ``Linear(d, V)`` whose ``weight`` is the *same* parameter as
        ``layer.weight`` -- an update to one is an update to the other.
    """
    num_embeddings, embedding_dim = layer.weight.shape
    readout = nn.Linear(embedding_dim, num_embeddings, bias=bias)
    # Tie: reuse the same Parameter rather than copy it. data_ptr stays shared.
    readout.weight = layer.weight
    return readout


def main() -> None:
    """Load the shared fixture, build the layer, and print the convention report."""
    from dlhub.embeddings.word_embeddings import lookup as reference_lookup
    from dlhub.embeddings.word_embeddings import make_fixture

    fixture = make_fixture()
    E, vocab = fixture["E"], fixture["vocab"]
    V = fixture["dims"]["V"]

    layer = embedding_from_reference(E)
    print(f"E is {E.shape} (d, V); layer.weight is {tuple(layer.weight.shape)} (V, d)")
    print(
        f"weight == E.T exactly: {np.array_equal(layer.weight.detach().numpy(), E.T)}"
    )
    print(
        f"from_pretrained froze the weight: requires_grad={layer.weight.requires_grad}"
    )

    gather_matches = all(
        np.array_equal(lookup(layer, i).numpy(), reference_lookup(E, i))
        for i in range(V)
    )
    print(f"gather == reference lookup for every word: {gather_matches}")

    readout = tied_readout(layer)
    shared = readout.weight.data_ptr() == layer.weight.data_ptr()
    print(f"tied read-out shares the embedding weight: {shared}")
    print(f"vocabulary: {vocab}")


if __name__ == "__main__":
    main()
