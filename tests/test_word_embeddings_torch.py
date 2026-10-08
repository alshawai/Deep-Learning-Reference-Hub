"""
Word Embeddings -- PyTorch Idiom Tests
======================================

The PyTorch port of the static word-embedding reference is an **idiom track**,
not a parity port: a lookup is a gather, so there is no nontrivial computation to
hold to a numeric tolerance. What these tests pin instead is the one thing a
reader must get right -- PyTorch's ``(V, d)`` row convention, the transpose of
Ng's ``(d, V)`` matrix -- and the ``nn.Embedding`` arguments that bite. Each check
is built to *fail* on a real mistake: a forgotten transpose, a dtype that rounds
the table, or the ``from_pretrained`` freeze default read the wrong way.

The anchors are *exact* rather than approximate on purpose. A gather copies
stored values and does no arithmetic, so when the fixture's ``float64`` dtype is
preserved the port's lookup equals the reference's byte-for-byte; the right
tolerance for "did the gather return the stored column" is equality, not ``atol``.

These tests live in a class whose name matches ``-k parity`` so the
cross-framework CI job -- the only job that installs ``torch`` -- selects and runs
them. Named any other way, the file would collect in the framework-free job, skip
at the ``importorskip`` below, and read green without ever executing. A skip is not
a pass here. (The name is a CI selector; this remains an idiom track, not a parity
port.)

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import numpy as np
import pytest

# Skip the whole module when torch is absent so the framework-free job can collect
# it. The cross-framework CI job installs torch and selects this file by its class
# name under ``-k parity``, so there every assertion below runs for real.
torch = pytest.importorskip("torch")

from dlhub.embeddings import word_embeddings as reference  # noqa: E402  (after gate)
from dlhub.pytorch.embeddings.word_embeddings import (  # noqa: E402  (after the gate)
    embedding_from_reference,
    lookup,
    tied_readout,
)


class TestWordEmbeddingParity:
    """
    The PyTorch idiom port's own-correctness anchors.

    The class name carries ``Parity`` only so the torch-installing CI job selects
    it (see the module docstring); the checks below are idiom-track correctness --
    the ``(V, d)`` convention and the ``nn.Embedding`` arguments -- not a numeric
    parity contract.
    """

    @pytest.fixture
    def toy(self):
        """The planted toy embedding matrix and its vocabulary, per the dossier."""
        return reference.make_fixture()

    @pytest.fixture
    def layer(self, toy):
        """The reference matrix loaded into a torch.nn.Embedding (weight == E.T)."""
        return embedding_from_reference(toy["E"])

    # --- The (V, d) transpose convention ---------------------------------

    def test_weight_is_the_reference_transpose(self, toy, layer):
        """
        PyTorch stores a word per row, so weight is (V, d) == E.T, exactly.

        Equality, not a tolerance: from_pretrained copies the table verbatim and
        the float64 dtype is preserved, so a single differing entry is a real bug
        -- a wrong transpose, or a dtype that rounded the table.
        """
        E = toy["E"]
        assert layer.weight.shape == (toy["dims"]["V"], toy["dims"]["d"])
        assert layer.weight.detach().numpy().dtype == np.float64
        assert np.array_equal(layer.weight.detach().numpy(), E.T)

    def test_an_untransposed_load_would_disagree(self, toy):
        """
        The transpose is load-bearing -- a discrimination control.

        Loading E without transposing builds a (d, V) layer whose row 0 is the
        royalty feature across words, not king's vector. Its gather must differ
        from the reference (here in both shape and value), or the convention test
        above would pass vacuously.
        """
        E = toy["E"]
        untransposed = torch.nn.Embedding.from_pretrained(torch.tensor(E))
        row0 = untransposed(torch.tensor([0]))[0].numpy()
        assert not np.array_equal(row0, reference.lookup(E, 0))

    # --- The lookup identity, torch side ---------------------------------

    def test_gather_matches_reference_lookup_for_every_word(self, toy, layer):
        """
        weight[i, :] == E[:, i] for every word: the gather is the reference lookup.

        This is the port's reason to exist -- the one-hot product e_w = E o_w done
        the way a real layer does it, as a gather by index. Exact for every word.
        """
        E, V, d = toy["E"], toy["dims"]["V"], toy["dims"]["d"]
        for i in range(V):
            gathered = lookup(layer, i).numpy()
            assert gathered.shape == (d,)
            assert np.array_equal(gathered, reference.lookup(E, i))

    def test_gather_equals_the_one_hot_product(self, toy, layer):
        """
        The spine, in torch: a one-hot row times the (V, d) weight is the gather.

        ``o_w @ weight`` selects row i, which is why one never materializes the
        product in practice -- the gather returns the same vector. This ties the
        reference's one_hot to the torch layer; exact because the one nonzero term
        is multiplied by 1.0 and the rest contribute exact zeros.
        """
        V = toy["dims"]["V"]
        weight = layer.weight.detach()
        for i in range(V):
            o = torch.tensor(reference.one_hot(i, V))  # (V,) float64, the reference's
            product = o @ weight  # (d,)
            assert torch.equal(product, lookup(layer, i))

    # --- nn.Embedding arguments that bite --------------------------------

    def test_from_pretrained_freezes_by_default(self, toy, layer):
        """
        from_pretrained defaults to freeze=True -- the opposite of nn.Embedding(V, d).

        The gotcha a reader fine-tuning pretrained vectors hits: the loaded table
        does not train unless freeze=False is passed.
        """
        assert layer.weight.requires_grad is False
        trainable = embedding_from_reference(toy["E"], freeze=False)
        assert trainable.weight.requires_grad is True

    def test_padding_idx_is_forwarded(self, toy):
        """
        padding_idx reaches the layer: the named row's gradient is held at zero.

        A practitioner's reason to pass it -- the padding token must not drift --
        so the argument is surfaced on the layer, not silently dropped.
        """
        layer = embedding_from_reference(toy["E"], freeze=False, padding_idx=0)
        assert layer.padding_idx == 0

    # --- Weight tying: the forward pointer, made concrete ----------------

    def test_tied_readout_shares_the_embedding_weight(self, layer):
        """
        A tied read-out is literally the same parameter, not a copy.

        Sharing storage is the whole point of weight tying: one (V, d) block
        serves as both the input embedding and the output projection. Equal
        ``data_ptr`` is the proof the tensor is shared, not duplicated.
        """
        readout = tied_readout(layer)
        assert readout.weight.shape == layer.weight.shape  # (V, d)
        assert readout.weight.data_ptr() == layer.weight.data_ptr()

    def test_tied_readout_scores_a_word_by_its_own_embedding(self, toy, layer):
        """
        With tied weights, the logit for word v is h . e_v.

        Linear(d, V) computes h @ weight.T and weight == E.T, so the score of word
        v is the dot product of the hidden state with word v's embedding -- the
        property that makes tying meaningful. atol=1e-12 (not equality) only
        because the matmul and the explicit dot product sum in different orders.
        """
        readout = tied_readout(layer)
        d, V = toy["dims"]["d"], toy["dims"]["V"]
        h = torch.arange(1.0, d + 1.0, dtype=torch.float64)  # a concrete hidden state
        logits = readout(h)
        expected = torch.stack([h @ lookup(layer, v) for v in range(V)])
        assert torch.allclose(logits, expected, atol=1e-12)
