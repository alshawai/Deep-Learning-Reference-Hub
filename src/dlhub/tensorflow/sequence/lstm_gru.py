"""
LSTM and GRU -- TensorFlow/Keras parity port
============================================

The hub's from-scratch gated cells, rewritten the way they are really written in
TensorFlow: a ``tf.keras.layers.LSTM`` or ``tf.keras.layers.GRU`` for the
recurrence, a ``tf.keras.layers.Dense`` plus ``softmax`` for the read-out, and
``tf.GradientTape`` standing in for the hand-derived backpropagation through
time of the reference modules.

Seven conventions of the Keras recurrent layers diverge from the hub reference
(Andrew Ng's Course 5 notation). Each is wiring a first Keras GRU or LSTM gets
wrong, so each is called out where it bites:

1. **``reset_after`` defaults to ``True``, and that is a different cell.** Keras
   defaults its GRU to the cuDNN-compatible form, which applies the reset gate
   *after* the recurrent transform:
   ``h_tilde = tanh(W_x x + r * (W_h h + b_h))``. The hub's reference -- Cho et
   al. (2014) and Ng -- applies it *before*, inside the concatenation ``W_c``
   multiplies. The two are not algebraically equal. Loading the reference
   weights into a ``reset_after=True`` layer reproduces nothing: on this topic's
   GRU fixture the hidden states differ by ``1.787`` in max absolute error,
   against the ``1e-8`` the ``reset_after=False`` layer meets to machine
   epsilon. :class:`GRUSequenceModel` therefore defaults to ``reset_after=False``
   -- the opposite of Keras -- and accepts the flag so the divergence can be
   demonstrated rather than described. (``torch.nn.GRU`` computes the
   reset-after form with no flag to change it, which is why this hub's framework
   parity track is TensorFlow.)
2. **The GRU's update gate is inverted.** Keras writes the blend as
   ``h = z * h_prev + (1 - z) * h_tilde``, putting the gate on the **old** state,
   where the reference writes ``a = Gamma_u * a_tilde + (1 - Gamma_u) * a_prev``.
   So ``z = 1 - Gamma_u``, and because both are logistic the pre-activation is
   simply negated: ``W_z = -W_u`` and ``b_z = -b_u``. The gradient carries the
   same flip, ``dW_u = -dW_z``. This is the ``-1.0`` in
   :attr:`GRUSequenceModel.GATE_SIGNS`, and it is the mistake that produces a
   plausible-looking cell that trains to the wrong thing. The LSTM needs no sign
   flips.
3. **Gates are packed into one matrix, in the framework's order.** Keras packs
   the GRU as ``[z, r, h]`` and the LSTM as ``[i, f, c, o]``; the hub teaches
   ``r, u, c`` and ``f, u, c, o``. Neither packing is the teaching order, so
   both are stated as data in :attr:`GatedSequenceModel.GATE_ORDER` rather than
   buried in slicing arithmetic.
4. **The weights are transposed, and split at ``n_a``.** Keras computes
   ``inputs @ kernel``, so its matrices are the transpose of the hub's
   ``(n_a, n_a + n_x)`` gate matrix -- and that matrix's two halves land in two
   different Keras weights: the recurrent half ``W_ga`` in ``recurrent_kernel``,
   the input half ``W_gx`` in ``kernel``. The hub's own
   :func:`~dlhub.nn.sequence._common.split_gate_matrix` makes the cut.
5. **``unit_forget_bias`` defaults to ``True``.** A freshly built Keras LSTM is
   not the paper's initialization: Keras adds ``1`` to the forget-gate slice of
   the bias, following Jozefowicz, Zaremba & Sutskever (2015), so the cell starts
   out remembering. It is a good default for training and an invisible one for
   anybody comparing against an equation. :class:`LSTMSequenceModel` passes
   ``unit_forget_bias=False`` so the layer before any weights are loaded is the
   plain cell the reference describes; loading reference weights overwrites the
   bias either way.
6. **Batch-first shapes.** Keras speaks ``(m, T_x, features)`` -- batch,
   sequence, feature -- while the hub reference speaks ``(features, m, T_x)``.
   :meth:`GatedSequenceModel.call` uses Keras's native order;
   :meth:`GatedSequenceModel.forward` and
   :meth:`GatedSequenceModel.bptt_gradients` bridge to the reference order so
   the parity test compares like with like.
7. **Only the *final* cell state is exposed.** ``return_state=True`` yields
   ``h^{<T_x>}`` and, for the LSTM, ``c^{<T_x>}`` -- never the sequence of cell
   states. The reference returns all of ``c`` because the additive cell path is
   the lesson; Keras treats it as internal. Forward parity on the cell state is
   therefore checked at the final step, which is what the layer offers.

References
----------
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural
  Computation*, 9(8), 1735-1780.
- Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). Learning to Forget:
  Continual Prediction with LSTM. *Neural Computation*, 12(10), 2451-2471.
- Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F.,
  Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN
  Encoder-Decoder for Statistical Machine Translation. EMNLP.
  https://arxiv.org/abs/1406.1078
- Jozefowicz, R., Zaremba, W., & Sutskever, I. (2015). An Empirical Exploration
  of Recurrent Network Architectures. ICML. (The forget-bias initialization
  Keras applies by default.)
- Ng, A. (2018). Deep Learning Specialization, Course 5 (Sequence Models),
  Week 1. https://www.coursera.org/specializations/deep-learning
- Keras. GRU layer. https://keras.io/api/layers/recurrent_layers/gru/
- Keras. LSTM layer. https://keras.io/api/layers/recurrent_layers/lstm/

Author
------
Deep Learning Reference Hub

License
-------
MIT
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from dlhub.nn.sequence._common import split_gate_matrix

__all__ = [
    "GRUSequenceModel",
    "GatedSequenceModel",
    "LSTMSequenceModel",
]


class GatedSequenceModel(keras.Model):
    """
    A Keras recurrent layer plus a softmax read-out, mapped onto hub notation.

    Holds everything the LSTM and the GRU ports share: the ``Dense`` read-out,
    the shape bridging between Keras's batch-first order and the reference's
    feature-first order, the graph-compiled gradient step, and the translation
    of Keras's packed gate weights to and from the reference's per-gate
    matrices. Each subclass supplies only its recurrent layer and the three
    class attributes that describe how that layer packs its gates.

    Subclasses are :class:`LSTMSequenceModel` and :class:`GRUSequenceModel`; the
    class is not abstract but is not useful on its own.

    Parameters
    ----------
    recurrent : keras.layers.Layer
        The recurrent layer, already configured with ``return_sequences=True``
        and ``return_state=True``.
    n_x : int
        Input dimension (features per timestep). Used to build the layers
        immediately, so reference weights can be loaded without a forward pass
        first.
    n_y : int
        Output dimension (number of classes).
    dtype : str, optional
        Compute dtype for every weight. Defaults to ``"float64"``, which the
        gradient tolerance requires; see the module ``Notes``.

    Attributes
    ----------
    GATE_ORDER : tuple[str, ...]
        The gates the framework packs into one weight matrix, named as the hub
        names them (``"f"``, ``"u"``, ``"c"``, ``"o"``, ``"r"``) and listed in
        the framework's packing order.
    GATE_SIGNS : tuple[float, ...]
        A sign per entry of ``GATE_ORDER``, applied to both the loaded weights
        and the returned gradients. Everything is ``1.0`` except Keras's
        inverted GRU update gate; see divergence 2 in the module docstring.
    STATE_NAMES : tuple[str, ...]
        The initial states this cell carries, named as the reference names them
        (``"a0"``, and ``"c0"`` for the LSTM). Fixes both the argument count of
        :meth:`bptt_gradients` and the keys of the state gradients it returns.
    recurrent : keras.layers.Layer
        The recurrent layer. Its ``kernel``, ``recurrent_kernel``, and ``bias``
        hold all of the reference's gate matrices, packed and transposed.
    readout : keras.layers.Dense
        The classifier's affine map, carrying ``W_y`` and ``b_y``.
    """

    GATE_ORDER: tuple[str, ...] = ()
    GATE_SIGNS: tuple[float, ...] = ()
    STATE_NAMES: tuple[str, ...] = ()

    def __init__(
        self,
        recurrent: keras.layers.Layer,
        n_x: int,
        n_y: int,
        dtype: str = "float64",
    ) -> None:
        super().__init__(dtype=dtype)
        self.recurrent = recurrent
        self.readout = keras.layers.Dense(n_y, dtype=dtype)
        # Built here rather than on first call so that load_reference_parameters
        # works on a fresh model. The timestep axis stays None: nothing in
        # either cell depends on T_x, and leaving it free is what lets the same
        # model run the LSTM fixture's T_x = 4 and any other length.
        self.recurrent.build((None, None, n_x))
        self.readout.build((None, self.recurrent.units))

    # --- Forward -------------------------------------------------------

    def call(
        self, inputs: tf.Tensor, initial_state: list[tf.Tensor] | None = None
    ) -> tuple[tf.Tensor, tf.Tensor, list[tf.Tensor]]:
        """
        Forward propagation through time, in Keras's native shape order.

        Parameters
        ----------
        inputs : tf.Tensor
            Input sequence of shape ``(m, T_x, n_x)`` -- batch, sequence,
            feature, which is Keras's order and the reverse of the reference's.
        initial_state : list[tf.Tensor] or None, optional
            Initial states, each ``(m, n_a)``: ``[a0]`` for the GRU and
            ``[a0, c0]`` for the LSTM. ``None`` gives the zero state, matching
            the reference convention ``a^{<0>} = 0``.

        Returns
        -------
        y_hat : tf.Tensor
            Per-timestep output distributions, shape ``(m, T_x, n_y)``.
        sequence : tf.Tensor
            All hidden states, shape ``(m, T_x, n_a)``.
        final_states : list[tf.Tensor]
            The states at ``t = T_x``, each ``(m, n_a)``, in the layer's own
            order. For the LSTM that is ``[a, c]`` -- the *only* place the cell
            state is exposed (divergence 7).
        """
        outputs = self.recurrent(inputs, initial_state=initial_state)
        sequence, final_states = outputs[0], list(outputs[1:])
        y_hat = keras.activations.softmax(self.readout(sequence), axis=-1)
        return y_hat, sequence, final_states

    # --- Weight mapping ------------------------------------------------

    def _pack_bias(self, flat: np.ndarray) -> np.ndarray:
        """
        Shape a flat per-gate bias vector the way this layer's ``bias`` expects.

        Parameters
        ----------
        flat : np.ndarray
            The gates' biases concatenated in ``GATE_ORDER``, shape
            ``(len(GATE_ORDER) * n_a,)``.

        Returns
        -------
        np.ndarray
            The same values, unchanged. Overridden by
            :class:`GRUSequenceModel`, whose reset-after form carries two bias
            rows.
        """
        return flat

    def load_reference_parameters(self, parameters: dict[str, np.ndarray]) -> None:
        """
        Copy the hub reference's weights into the Keras layers, in place.

        Performs the whole mapping the module docstring describes: each
        reference gate matrix ``W_g`` of shape ``(n_a, n_a + n_x)`` is signed
        (divergence 2), split at column ``n_a`` into its recurrent and input
        halves (divergence 4), transposed, and concatenated with the other gates
        in the framework's packing order (divergence 3). The read-out's ``W_y``
        is transposed into the ``Dense`` kernel.

        Parameters
        ----------
        parameters : dict[str, np.ndarray]
            The reference weights for this cell, named as the reference names
            them -- ``Wf``/``Wu``/``Wc``/``Wo`` or ``Wr``/``Wu``/``Wc``, the
            matching ``b`` vectors, plus ``Wy`` ``(n_y, n_a)`` and ``by``
            ``(n_y, 1)``.
        """
        n_a = self.recurrent.units
        kernel_blocks, recurrent_blocks, bias_blocks = [], [], []

        for gate, sign in zip(self.GATE_ORDER, self.GATE_SIGNS, strict=True):
            recurrent_half, input_half = split_gate_matrix(
                sign * parameters[f"W{gate}"], n_a
            )
            recurrent_blocks.append(recurrent_half.T)
            kernel_blocks.append(input_half.T)
            bias_blocks.append((sign * parameters[f"b{gate}"]).reshape(-1))

        self.recurrent.set_weights(
            [
                np.concatenate(kernel_blocks, axis=1),  # (n_x, g * n_a)
                np.concatenate(recurrent_blocks, axis=1),  # (n_a, g * n_a)
                self._pack_bias(np.concatenate(bias_blocks)),
            ]
        )
        self.readout.set_weights(
            [parameters["Wy"].T, parameters["by"].reshape(-1)]  # (n_a, n_y), (n_y,)
        )

    def _unpack_gate_gradients(
        self, d_kernel: np.ndarray, d_recurrent: np.ndarray, d_bias: np.ndarray
    ) -> dict[str, np.ndarray]:
        """
        Invert :meth:`load_reference_parameters` on a gradient.

        Slices the packed gradients back into one ``(n_a, n_a + n_x)`` matrix
        per gate, re-joining the recurrent and input halves in the reference's
        column order and re-applying each gate's sign.

        Parameters
        ----------
        d_kernel : np.ndarray
            Gradient w.r.t. ``kernel``, shape ``(n_x, g * n_a)``.
        d_recurrent : np.ndarray
            Gradient w.r.t. ``recurrent_kernel``, shape ``(n_a, g * n_a)``.
        d_bias : np.ndarray
            Gradient w.r.t. ``bias``, shape ``(g * n_a,)``.

        Returns
        -------
        dict[str, np.ndarray]
            ``dW<gate>`` and ``db<gate>`` for every gate in ``GATE_ORDER``,
            keyed and shaped exactly as the reference's BPTT returns them.
        """
        n_a = self.recurrent.units
        gradients: dict[str, np.ndarray] = {}

        for index, (gate, sign) in enumerate(
            zip(self.GATE_ORDER, self.GATE_SIGNS, strict=True)
        ):
            block = slice(index * n_a, (index + 1) * n_a)
            gradients[f"dW{gate}"] = sign * np.concatenate(
                [d_recurrent[:, block].T, d_kernel[:, block].T], axis=1
            )
            gradients[f"db{gate}"] = sign * d_bias[block].reshape(-1, 1)

        return gradients

    # --- Shape bridging ------------------------------------------------

    def _to_keras(
        self, x: np.ndarray, states: list[np.ndarray]
    ) -> tuple[tf.Tensor, list[tf.Tensor]]:
        """
        Convert reference-shaped arrays to Keras tensors (divergence 6).

        Parameters
        ----------
        x : np.ndarray
            Input sequence, shape ``(n_x, m, T_x)``.
        states : list[np.ndarray]
            Initial states, each ``(n_a, m)``.

        Returns
        -------
        x_tensor : tf.Tensor
            Shape ``(m, T_x, n_x)``.
        state_tensors : list[tf.Tensor]
            Each ``(m, n_a)``.

        Raises
        ------
        ValueError
            If the number of states does not match :attr:`STATE_NAMES`.
        """
        if len(states) != len(self.STATE_NAMES):
            raise ValueError(
                f"{type(self).__name__} carries the initial states "
                f"{self.STATE_NAMES}; got {len(states)}"
            )
        dtype = self.recurrent.compute_dtype
        x_tensor = tf.convert_to_tensor(np.transpose(x, (1, 2, 0)), dtype=dtype)
        state_tensors = [
            tf.convert_to_tensor(np.ascontiguousarray(state.T), dtype=dtype)
            for state in states
        ]
        return x_tensor, state_tensors

    @staticmethod
    def _to_reference(sequence: tf.Tensor) -> np.ndarray:
        """
        Convert a Keras ``(m, T_x, features)`` sequence to ``(features, m, T_x)``.

        Parameters
        ----------
        sequence : tf.Tensor
            Any per-timestep Keras output.

        Returns
        -------
        np.ndarray
            The same values in the reference's feature-first order.
        """
        return np.transpose(np.asarray(sequence), (2, 0, 1))

    # --- Loss and gradients --------------------------------------------

    @tf.function
    def loss_and_gradients(
        self, x: tf.Tensor, y: tf.Tensor, initial_state: list[tf.Tensor]
    ) -> tuple:
        """
        One graph-compiled forward-and-backward step, in Keras shape order.

        Forms the reference's total loss ``-(y * log(y_hat)).sum() / m``
        (cross-entropy summed over timesteps, averaged over the batch) and
        differentiates it with ``tf.GradientTape``. The tape watches the input
        and the initial states as well as the weights, because the reference's
        BPTT returns ``dx`` and ``da0`` (and ``dc0``) and parity is claimed on
        those too.

        Parameters
        ----------
        x : tf.Tensor
            Input sequence, shape ``(m, T_x, n_x)``.
        y : tf.Tensor
            One-hot targets, shape ``(m, T_x, n_y)``.
        initial_state : list[tf.Tensor]
            Initial states, each ``(m, n_a)``.

        Returns
        -------
        loss : tf.Tensor
            The scalar total loss.
        y_hat : tf.Tensor
            Output distributions, shape ``(m, T_x, n_y)``.
        sequence : tf.Tensor
            Hidden states, shape ``(m, T_x, n_a)``.
        final_states : list[tf.Tensor]
            States at ``t = T_x``.
        gradients : list[tf.Tensor]
            Gradients w.r.t. ``kernel``, ``recurrent_kernel``, ``bias``, the
            read-out's kernel and bias, ``x``, and each initial state, in that
            order.
        """
        # Unpacking by name rather than indexing trainable_weights documents the
        # order set_weights uses, and fails loudly if Keras ever adds a weight.
        kernel, recurrent_kernel, bias = self.recurrent.weights
        readout_kernel, readout_bias = self.readout.weights

        with tf.GradientTape() as tape:
            tape.watch([x, *initial_state])
            y_hat, sequence, final_states = self(x, initial_state=initial_state)
            # The batch is axis 0 in Keras order; the reference averages over it
            # and sums over time. Matching the loss exactly is the precondition
            # for matching gradients.
            m = tf.cast(tf.shape(x)[0], y_hat.dtype)
            loss = -tf.reduce_sum(y * tf.math.log(y_hat)) / m

        sources = [
            kernel,
            recurrent_kernel,
            bias,
            readout_kernel,
            readout_bias,
            x,
            *initial_state,
        ]
        return loss, y_hat, sequence, final_states, tape.gradient(loss, sources)

    # --- The reference-facing entry points -----------------------------

    def forward(self, x: np.ndarray, *states: np.ndarray) -> dict[str, np.ndarray]:
        """
        Run the forward pass on reference-shaped arrays, eagerly.

        Parameters
        ----------
        x : np.ndarray
            Input sequence, shape ``(n_x, m, T_x)``.
        *states : np.ndarray
            The initial states this cell carries, each ``(n_a, m)``, in
            :attr:`STATE_NAMES` order: ``a0`` for the GRU, then ``c0`` for the
            LSTM.

        Returns
        -------
        dict[str, np.ndarray]
            ``a`` ``(n_a, m, T_x)``, ``y_pred`` ``(n_y, m, T_x)``, and one
            ``<name>_final`` entry per state -- ``a_final`` and, for the LSTM,
            ``c_final``, each ``(n_a, m)``. The cell-state *sequence* has no
            Keras equivalent (divergence 7).
        """
        x_tensor, state_tensors = self._to_keras(x, list(states))
        y_hat, sequence, final_states = self(x_tensor, initial_state=state_tensors)

        result = {
            "a": self._to_reference(sequence),
            "y_pred": self._to_reference(y_hat),
        }
        for name, final in zip(self.STATE_NAMES, final_states, strict=True):
            result[f"{name[0]}_final"] = np.asarray(final).T
        return result

    def bptt_gradients(
        self, x: np.ndarray, y: np.ndarray, *states: np.ndarray
    ) -> dict[str, np.ndarray]:
        """
        Autograd BPTT, returned in the hub reference's names and shapes.

        Runs :meth:`loss_and_gradients` on the bridged tensors and unpacks every
        result back into the reference's convention, so the parity test compares
        like with like rather than reasoning about packing at the assertion.

        Parameters
        ----------
        x : np.ndarray
            Input sequence, shape ``(n_x, m, T_x)``.
        y : np.ndarray
            One-hot targets, shape ``(n_y, m, T_x)``.
        *states : np.ndarray
            The initial states this cell carries, each ``(n_a, m)``, in
            :attr:`STATE_NAMES` order.

        Returns
        -------
        dict[str, np.ndarray]
            The tape's gradients keyed as the reference names them --
            ``dW<gate>`` and ``db<gate>`` for every gate, ``dWy``, ``dby``,
            ``dx`` ``(n_x, m, T_x)``, and ``da0`` (plus ``dc0`` for the LSTM) --
            together with the scalar ``loss`` and the forward ``a`` and
            ``y_pred`` for the forward-parity check.
        """
        x_tensor, state_tensors = self._to_keras(x, list(states))
        dtype = self.recurrent.compute_dtype
        y_tensor = tf.convert_to_tensor(np.transpose(y, (1, 2, 0)), dtype=dtype)

        loss, y_hat, sequence, _, gradients = self.loss_and_gradients(
            x_tensor, y_tensor, state_tensors
        )
        d_kernel, d_recurrent, d_bias, d_readout, d_readout_bias = (
            np.asarray(g) for g in gradients[:5]
        )
        d_x, *d_states = (np.asarray(g) for g in gradients[5:])

        result = self._unpack_gate_gradients(d_kernel, d_recurrent, d_bias)
        result["dWy"] = d_readout.T
        result["dby"] = d_readout_bias.reshape(-1, 1)
        result["dx"] = np.transpose(d_x, (2, 0, 1))
        for name, d_state in zip(self.STATE_NAMES, d_states, strict=True):
            result[f"d{name}"] = d_state.T

        result["loss"] = float(loss)
        result["a"] = self._to_reference(sequence)
        result["y_pred"] = self._to_reference(y_hat)
        return result


class LSTMSequenceModel(GatedSequenceModel):
    """
    The hub's LSTM, written the Keras way.

    A ``tf.keras.layers.LSTM`` carries the recurrence and a ``Dense`` +
    ``softmax`` forms the per-timestep read-out ``y_hat``. Keras packs the four
    gates as ``[i, f, c, o]``; the hub's update gate ``Gamma_u`` *is* Hochreiter
    & Schmidhuber's input gate ``i``, which is why :attr:`GATE_ORDER` opens with
    ``"u"``. No gate needs a sign flip -- unlike the GRU, Keras's LSTM blend is
    written the same way round as the reference's.

    Parameters
    ----------
    n_x : int
        Input dimension (features per timestep).
    n_a : int
        Hidden-state dimension.
    n_y : int
        Output dimension (number of classes).
    dtype : str, optional
        Compute dtype. Defaults to ``"float64"`` for the gradient check.
    """

    # Keras packs [i, f, c, o]. The hub writes the input gate Gamma_u, so the
    # first block is "u"; the rest is the hub's own forget, candidate, output.
    GATE_ORDER = ("u", "f", "c", "o")
    GATE_SIGNS = (1.0, 1.0, 1.0, 1.0)
    STATE_NAMES = ("a0", "c0")

    def __init__(self, n_x: int, n_a: int, n_y: int, dtype: str = "float64") -> None:
        super().__init__(
            keras.layers.LSTM(
                n_a,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=True,
                return_state=True,
                # Keras defaults this to True, adding 1 to the forget-gate bias
                # slice at initialization. Off here so a freshly built layer is
                # the plain cell the reference describes (divergence 5).
                unit_forget_bias=False,
                dtype=dtype,
            ),
            n_x,
            n_y,
            dtype=dtype,
        )


class GRUSequenceModel(GatedSequenceModel):
    """
    The hub's GRU, written the Keras way -- with the two flags that matter.

    A ``tf.keras.layers.GRU`` carries the recurrence and a ``Dense`` +
    ``softmax`` forms the read-out. Two of Keras's conventions have to be
    reconciled before this is the reference's cell at all, and both are the
    module docstring's divergences 1 and 2: ``reset_after`` must be ``False``,
    and the update gate is stored inverted, so its weight, bias, and gradient
    all carry a minus sign.

    Parameters
    ----------
    n_x : int
        Input dimension (features per timestep).
    n_a : int
        Hidden-state dimension.
    n_y : int
        Output dimension (number of classes).
    reset_after : bool, optional
        Where the reset gate is applied. ``False`` (the default *here*, and the
        opposite of Keras's own default) is the Cho/Ng cell the hub reference
        derives. ``True`` is the cuDNN-compatible reset-after cell, a different
        function of the same weights; it is accepted so the divergence can be
        measured rather than asserted in prose, and :meth:`bptt_gradients`
        refuses it.
    dtype : str, optional
        Compute dtype. Defaults to ``"float64"`` for the gradient check.
    """

    # Keras packs [z, r, h]; z = 1 - Gamma_u, hence the -1.0 (divergence 2).
    GATE_ORDER = ("u", "r", "c")
    GATE_SIGNS = (-1.0, 1.0, 1.0)
    STATE_NAMES = ("a0",)

    def __init__(
        self,
        n_x: int,
        n_a: int,
        n_y: int,
        reset_after: bool = False,
        dtype: str = "float64",
    ) -> None:
        super().__init__(
            keras.layers.GRU(
                n_a,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=True,
                return_state=True,
                reset_after=reset_after,
                dtype=dtype,
            ),
            n_x,
            n_y,
            dtype=dtype,
        )

    def _pack_bias(self, flat: np.ndarray) -> np.ndarray:
        """
        Shape the per-gate biases for this layer's ``reset_after`` setting.

        ``reset_after=False`` takes a single bias row, exactly the reference's.
        ``reset_after=True`` takes two -- an input bias and a separate recurrent
        bias, which is what lets the cuDNN kernel apply the reset gate after the
        recurrent matmul. The reference has no second bias, so it is zeroed:
        that makes the comparison in the divergence test a clean one, with only
        the flag changed.

        Parameters
        ----------
        flat : np.ndarray
            The gates' biases concatenated in ``GATE_ORDER``, shape ``(3 n_a,)``.

        Returns
        -------
        np.ndarray
            Shape ``(3 n_a,)`` when ``reset_after=False``, else ``(2, 3 n_a)``.
        """
        if not self.recurrent.reset_after:
            return flat
        return np.stack([flat, np.zeros_like(flat)])

    def bptt_gradients(
        self, x: np.ndarray, y: np.ndarray, *states: np.ndarray
    ) -> dict[str, np.ndarray]:
        """
        Autograd BPTT -- available only for the reference's cell.

        Parameters
        ----------
        x : np.ndarray
            Input sequence, shape ``(n_x, m, T_x)``.
        y : np.ndarray
            One-hot targets, shape ``(n_y, m, T_x)``.
        *states : np.ndarray
            The initial hidden state ``a0``, shape ``(n_a, m)``.

        Returns
        -------
        dict[str, np.ndarray]
            As :meth:`GatedSequenceModel.bptt_gradients`.

        Raises
        ------
        ValueError
            If the model was built with ``reset_after=True``. That cell is a
            different function, so there is no reference gradient its gradients
            could be compared against; its two bias rows cannot be unpacked into
            the reference's single ``db`` either.
        """
        if self.recurrent.reset_after:
            raise ValueError(
                "reset_after=True is the cuDNN reset-after cell, not the "
                "Cho/Ng cell dlhub.nn.sequence.gru derives, so its gradients "
                "have no reference gradients to be compared against. Build the "
                "model with reset_after=False for parity."
            )
        return super().bptt_gradients(x, y, *states)


def main() -> None:
    """Load both shared fixtures, run the ports, and print a parity report."""
    from dlhub.nn.sequence.gru import gru_backward, gru_forward
    from dlhub.nn.sequence.gru import make_fixture as make_gru_fixture
    from dlhub.nn.sequence.lstm import lstm_backward, lstm_forward
    from dlhub.nn.sequence.lstm import make_fixture as make_lstm_fixture
    from dlhub.nn.sequence.rnn import compute_loss

    def report(title, reference, port, keys):
        print(f"\n{title}")
        print(
            f"  loss (reference / port): {reference['loss']:.10f} / {port['loss']:.10f}"
        )
        for key in keys:
            analytic, autograd = reference[key], port[key]
            numerator = np.linalg.norm(analytic - autograd)
            denominator = np.linalg.norm(analytic) + np.linalg.norm(autograd)
            relative = 0.0 if denominator == 0 else numerator / denominator
            print(f"  {key:>5}: relative error {relative:.2e}")

    # --- LSTM ---------------------------------------------------------
    fixture = make_lstm_fixture()
    x, a0, c0, parameters, y = (
        fixture["x"],
        fixture["a0"],
        fixture["c0"],
        fixture["parameters"],
        fixture["y"],
    )
    dims = fixture["dims"]

    _, _, y_reference, caches = lstm_forward(x, parameters, a0, c0)
    reference = lstm_backward(y, caches, parameters)
    reference["loss"] = compute_loss(y_reference, y)

    model = LSTMSequenceModel(dims["n_x"], dims["n_a"], dims["n_y"])
    model.load_reference_parameters(parameters)
    report(
        "LSTM (tf.keras.layers.LSTM)",
        reference,
        model.bptt_gradients(x, y, a0, c0),
        ("dWf", "dWu", "dWc", "dWo", "dWy", "dby", "da0", "dc0", "dx"),
    )

    # --- GRU ----------------------------------------------------------
    fixture = make_gru_fixture()
    x, a0, parameters, y = (
        fixture["x"],
        fixture["a0"],
        fixture["parameters"],
        fixture["y"],
    )
    dims = fixture["dims"]

    a_reference, y_reference, caches = gru_forward(x, parameters, a0)
    reference = gru_backward(y, caches, parameters)
    reference["loss"] = compute_loss(y_reference, y)

    model = GRUSequenceModel(dims["n_x"], dims["n_a"], dims["n_y"])
    model.load_reference_parameters(parameters)
    report(
        "GRU (tf.keras.layers.GRU, reset_after=False)",
        reference,
        model.bptt_gradients(x, y, a0),
        ("dWr", "dWu", "dWc", "dWy", "dby", "da0", "dx"),
    )

    # --- The divergence, measured -------------------------------------
    # Same weights, same fixture, one flag changed. Keras's own default is the
    # one that does not reproduce the reference (divergence 1).
    reset_after = GRUSequenceModel(
        dims["n_x"], dims["n_a"], dims["n_y"], reset_after=True
    )
    reset_after.load_reference_parameters(parameters)
    mismatch = np.max(np.abs(reset_after.forward(x, a0)["a"] - a_reference))
    print(f"\nGRU reset_after=True vs. the reference: max abs error {mismatch:.4f}")


if __name__ == "__main__":
    main()
