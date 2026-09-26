"""
Character-Level Language Model and Sampling -- PyTorch (canonical)
==================================================================

The hub's character-level recurrent language model, written the way it is really
written in PyTorch: a ``torch.nn.RNN`` tanh cell and an ``nn.Linear`` softmax
read-out, trained with ``F.cross_entropy`` and ``torch.nn.utils.clip_grad_norm_``,
and sampled autoregressively with ``torch.multinomial`` from a seeded
``torch.Generator``.

This is the topic's **canonical** implementation, not a parity port: the language
model introduces no new from-scratch differentiable mathematics (the vanilla
RNN cell's BPTT is owned by :mod:`dlhub.nn.sequence.rnn`), so there is no NumPy
reference to reproduce. The module is therefore checked on its **own
correctness** -- the initialisation anchor, a strictly decreasing training loss,
reproducible sampling, a shuffled-target discrimination control, and a clip
cross-check -- rather than against a from-scratch module. See
``tests/test_language_model.py``.

The equations this module honours are stated in the topic's reference view
(``docs/reference/language-modeling-and-sampling.md``): the autoregressive
factorisation, the shifted-target per-character cross-entropy, perplexity and the
``ln V`` anchor, temperature-scaled sampling, and global-norm clipping.

References
----------
- Ng, A. (2018). Deep Learning Specialization, Course 5 (Sequence Models),
  Week 1. https://www.coursera.org/specializations/deep-learning
- Graves, A. (2013). Generating Sequences With Recurrent Neural Networks.
  https://arxiv.org/abs/1308.0850
- Karpathy, A. (2015). The Unreasonable Effectiveness of Recurrent Neural
  Networks. https://karpathy.github.io/2015/05/21/rnn-effectiveness/
- Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training
  Recurrent Neural Networks. ICML. https://arxiv.org/abs/1211.5063

Author
------
Deep Learning Reference Hub

License
-------
MIT

Notes
-----
- **The loss is** ``F.cross_entropy`` **on raw logits.** It applies a natural-log
  ``log_softmax`` internally with a mean reduction over tokens, which is exactly
  the per-character cross-entropy in nats the topic reports -- so perplexity is
  ``exp`` of it. The read-out returns *logits*, never a softmax; the softmax
  lives inside the loss (training) and is applied explicitly only when sampling.
- **The** ``ln V`` **anchor rides on PyTorch's default initialisation.**
  ``nn.RNN`` and ``nn.Linear`` initialise their weights uniformly in
  ``(-1/sqrt(H), 1/sqrt(H))``, small enough that the untrained read-out is
  approximately uniform over the ``V`` characters, so the per-character
  cross-entropy starts near ``ln V``. This is the framework's own initialisation,
  not the paper's, and it is what makes the anchor hold without any manual setup.
- **clip_grad_norm_ is not exactly** ``max_norm / ||g||``. PyTorch computes the
  clip coefficient as ``max_norm / (||g|| + 1e-6)`` -- a fixed ``1e-6`` guards
  the division against a zero gradient. The from-scratch ``clip_gradients`` uses
  ``max_norm / ||g||`` with no such term, so on ordinary-scale gradients the two
  agree only to roughly ``1e-6 / ||g||`` (about ``1e-8`` here), not to float64
  machine precision. The underlying global-norm rule is identical; the epsilon is
  the whole difference, and it vanishes as ``||g||`` grows. A practitioner
  reaching for ``clip_grad_norm_`` should know it carries this safety term.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

# The fixture corpus: a fixed literal, roughly ten short lowercase words, one per
# line and newline-terminated, so the vocabulary and every encoding are pinned
# and cannot drift. It is never read from an external file.
CORPUS = "cat\ndog\nfish\nbird\nfrog\nbear\nlion\nwolf\ndeer\nhawk\n"

# The fixture's hidden width and seeds (see the dossier's Fixture section).
HIDDEN_SIZE = 32
WEIGHT_SEED = 1
SAMPLE_SEED = 1


class CharRNN(nn.Module):
    """
    Character-level vanilla-RNN language model.

    A ``torch.nn.RNN`` (tanh nonlinearity) carries the recurrence
    ``a^{<t>} = tanh(W_ax x^{<t>} + W_aa a^{<t-1>} + b_a)`` and an ``nn.Linear``
    forms the read-out logits ``z^{<t>} = W_ya a^{<t>} + b_y``. The softmax that
    turns those logits into the next-character distribution is *not* part of the
    module: at training time it is folded into :func:`corpus_cross_entropy`
    through ``F.cross_entropy``, and at sampling time it is applied explicitly in
    :func:`sample`. Keeping the module logit-valued is the idiomatic PyTorch shape
    and is what lets the loss stay numerically stable.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size ``V``. Because the model is character-level and every
        token is a one-hot column, this is both the input and the output width
        (``n_x = n_y = V``).
    hidden_size : int
        Hidden-state dimension ``n_a``.

    Attributes
    ----------
    rnn : torch.nn.RNN
        The tanh recurrence, in PyTorch's default ``(T, batch, feature)`` shape
        order.
    readout : torch.nn.Linear
        The affine read-out carrying ``W_ya`` and ``b_y``.
    """

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.rnn = nn.RNN(input_size=vocab_size, hidden_size=hidden_size)
        self.readout = nn.Linear(hidden_size, vocab_size)

    def forward(
        self, x: torch.Tensor, a0: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward propagation through time, returning read-out logits.

        Parameters
        ----------
        x : torch.Tensor
            One-hot input sequence of shape ``(T, batch, V)`` -- sequence, batch,
            feature (PyTorch's default, ``batch_first=False``).
        a0 : torch.Tensor or None, optional
            Initial hidden state of shape ``(1, batch, n_a)``. Defaults to
            ``None``, which ``nn.RNN`` treats as the zero state, matching the
            convention ``a^{<0>} = 0``.

        Returns
        -------
        logits : torch.Tensor
            Per-timestep read-out logits ``z^{<t>}``, shape ``(T, batch, V)``.
        a_n : torch.Tensor
            The final hidden state ``a^{<T>}``, shape ``(1, batch, n_a)``, for
            threading the recurrence across autoregressive sampling steps.
        """
        a, a_n = self.rnn(x, a0)
        logits = self.readout(a)
        return logits, a_n


def build_vocabulary(corpus: str) -> tuple[list[str], dict[str, int], dict[int, str]]:
    r"""
    Build the deterministic character vocabulary from a corpus.

    The vocabulary is the **sorted** set of unique characters, including the
    terminating newline ``\n``. Sorting is what makes the encoding identical on
    every machine -- an unordered ``set`` iteration order would silently change
    the index maps and every one-hot column with them.

    Parameters
    ----------
    corpus : str
        The training text.

    Returns
    -------
    vocab : list[str]
        The sorted unique characters, ``V = len(vocab)``.
    char_to_ix : dict[str, int]
        Character to index, by sorted order.
    ix_to_char : dict[int, str]
        The inverse map.
    """
    vocab = sorted(set(corpus))
    char_to_ix = {char: index for index, char in enumerate(vocab)}
    ix_to_char = {index: char for index, char in enumerate(vocab)}
    return vocab, char_to_ix, ix_to_char


def encode_word(
    word: str, char_to_ix: dict[str, int], vocab_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Encode one word as a shifted (input, target) training pair.

    The target sequence is the word's characters followed by the terminating
    newline, ``[c_1, ..., c_k, \n]``. The input at each step is the *previous*
    character as a one-hot column, with the first input the **zero vector**
    (there is no character before ``c_1``). So step ``t`` sees ``x^{<t>}`` and is
    trained to predict ``y^{<t>} = x^{<t+1>}``.

    Parameters
    ----------
    word : str
        A single word, without its trailing newline.
    char_to_ix : dict[str, int]
        The vocabulary index map (see :func:`build_vocabulary`).
    vocab_size : int
        The vocabulary size ``V``, the width of each one-hot input column.

    Returns
    -------
    inputs : torch.Tensor
        One-hot inputs of shape ``(T, 1, V)``, batch size one; ``inputs[0]`` is
        the zero vector ``x^{<1>}``.
    targets : torch.Tensor
        Target character indices of shape ``(T,)``, ending in the newline index.
    """
    target_ids = [char_to_ix[char] for char in word] + [char_to_ix["\n"]]
    targets = torch.tensor(target_ids, dtype=torch.long)

    seq_len = len(target_ids)
    inputs = torch.zeros(seq_len, 1, vocab_size)
    for step in range(1, seq_len):
        # x^{<t>} is the one-hot of the previous character, target_ids[t-1].
        inputs[step, 0, target_ids[step - 1]] = 1.0
    return inputs, targets


def make_fixture() -> dict:
    """
    Build the deterministic fixture the module and its tests share.

    The corpus is the committed :data:`CORPUS` literal, so the vocabulary and
    every encoding are pinned. Each word becomes its own ``(inputs, targets)``
    pair processed one at a time -- no padding or masking, the
    Dinosaur-Island-shaped choice the dossier fixes.

    Returns
    -------
    dict
        ``{"corpus", "vocab", "char_to_ix", "ix_to_char", "vocab_size",
        "hidden_size", "examples", "weight_seed", "sample_seed"}`` where
        ``examples`` is the list of per-word ``(inputs, targets)`` pairs.
    """
    vocab, char_to_ix, ix_to_char = build_vocabulary(CORPUS)
    vocab_size = len(vocab)
    words = [word for word in CORPUS.split("\n") if word]
    examples = [encode_word(word, char_to_ix, vocab_size) for word in words]
    return {
        "corpus": CORPUS,
        "vocab": vocab,
        "char_to_ix": char_to_ix,
        "ix_to_char": ix_to_char,
        "vocab_size": vocab_size,
        "hidden_size": HIDDEN_SIZE,
        "examples": examples,
        "weight_seed": WEIGHT_SEED,
        "sample_seed": SAMPLE_SEED,
    }


def build_model(
    vocab_size: int, hidden_size: int = HIDDEN_SIZE, *, seed: int = WEIGHT_SEED
) -> CharRNN:
    """
    Construct a :class:`CharRNN` with reproducible initial weights.

    Seeds ``torch.manual_seed`` immediately before constructing the module, so
    the PyTorch default initialisation (uniform in ``(-1/sqrt(n_a), 1/sqrt(n_a))``)
    is drawn deterministically. This is what pins the ``ln V`` initialisation
    anchor to a repeatable value.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size ``V``.
    hidden_size : int, optional
        Hidden width ``n_a``. Defaults to :data:`HIDDEN_SIZE`.
    seed : int, optional
        Seed for ``torch.manual_seed``. Defaults to :data:`WEIGHT_SEED`.

    Returns
    -------
    CharRNN
        A freshly initialised model.
    """
    torch.manual_seed(seed)
    return CharRNN(vocab_size, hidden_size)


def corpus_cross_entropy(
    model: CharRNN, examples: list[tuple[torch.Tensor, torch.Tensor]]
) -> torch.Tensor:
    """
    Per-character cross-entropy over the whole corpus, in nats.

    Runs every word (each with its own zero initial state -- one word at a time)
    and pools all predicted tokens, then takes ``F.cross_entropy`` with its
    default mean reduction. Because ``F.cross_entropy`` uses the natural
    logarithm and averages over tokens, the result is exactly the topic's
    per-character cross-entropy: the family's summed-over-time, batch-mean loss
    divided by the sequence length.

    Parameters
    ----------
    model : CharRNN
        The language model.
    examples : list[tuple[torch.Tensor, torch.Tensor]]
        Per-word ``(inputs, targets)`` pairs from :func:`encode_word`.

    Returns
    -------
    torch.Tensor
        A scalar tensor: the mean cross-entropy over every predicted character.
    """
    all_logits = []
    all_targets = []
    for inputs, targets in examples:
        logits, _ = model(inputs)  # (T, 1, V)
        all_logits.append(logits.reshape(-1, logits.shape[-1]))  # (T, V)
        all_targets.append(targets)  # (T,)
    return F.cross_entropy(torch.cat(all_logits), torch.cat(all_targets))


def perplexity(cross_entropy: float | torch.Tensor) -> float:
    """
    Perplexity, the exponential of the per-character cross-entropy.

    Parameters
    ----------
    cross_entropy : float or torch.Tensor
        A per-character cross-entropy in nats.

    Returns
    -------
    float
        ``exp(cross_entropy)`` -- the model's effective branching factor. At
        initialisation this is ``~ V``; its floor is ``1``.
    """
    return math.exp(float(cross_entropy))


def train(
    model: CharRNN,
    examples: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    steps: int,
    learning_rate: float = 0.1,
    max_norm: float = 5.0,
) -> list[float]:
    """
    Full-corpus gradient descent with global-norm gradient clipping.

    Each step evaluates the per-character cross-entropy over the whole corpus,
    backpropagates, clips the global gradient norm with
    ``torch.nn.utils.clip_grad_norm_`` (the exploding-gradient remedy), and takes
    one SGD step. The loss for each step is recorded before the update.

    Parameters
    ----------
    model : CharRNN
        The model to train, updated in place.
    examples : list[tuple[torch.Tensor, torch.Tensor]]
        Per-word ``(inputs, targets)`` pairs.
    steps : int
        Number of full-corpus gradient-descent steps.
    learning_rate : float, optional
        SGD step size. Defaults to ``0.1`` -- small enough that full-corpus
        descent on the fixture is strictly monotonic, which the training-smoke
        test relies on; a larger rate converges faster but can overshoot and
        raise the loss on a step.
    max_norm : float, optional
        The global-norm clipping threshold. Defaults to ``5.0``.

    Returns
    -------
    list[float]
        The per-character cross-entropy at each step, before that step's update.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    history = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = corpus_cross_entropy(model, examples)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        history.append(float(loss.detach()))
    return history


def sample(
    model: CharRNN,
    char_to_ix: dict[str, int],
    ix_to_char: dict[int, str],
    *,
    generator: torch.Generator,
    temperature: float = 1.0,
    max_length: int = 50,
) -> str:
    """
    Generate a novel string autoregressively, stochastically, and reproducibly.

    Starting from the zero input ``x^{<1>}``, each step draws the next character
    from ``softmax(z^{<t>} / temperature)`` with ``torch.multinomial`` using the
    explicitly-passed ``generator``, feeds the drawn character back as the next
    one-hot input, and threads the hidden state forward. Generation stops when
    the newline is drawn or ``max_length`` characters have been emitted (the cap
    guarantees termination even from an untrained model).

    Parameters
    ----------
    model : CharRNN
        The model to sample from.
    char_to_ix, ix_to_char : dict
        The vocabulary index maps.
    generator : torch.Generator
        A seeded generator; the same seed yields a byte-for-byte identical
        string, which is the whole point of passing it explicitly.
    temperature : float, optional
        ``T > 0``. ``T = 1`` is the model's own distribution, ``T -> 0+`` is
        greedy, ``T > 1`` diversifies. Defaults to ``1.0``.
    max_length : int, optional
        The hard cap ``L`` on emitted characters. Defaults to ``50``.

    Returns
    -------
    str
        The sampled string, including the terminating newline when one is drawn.

    Raises
    ------
    ValueError
        If ``temperature`` is not strictly positive (``z / T`` would divide by
        zero or flip the distribution).
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0; got {temperature}")

    vocab_size = len(char_to_ix)
    newline_ix = char_to_ix["\n"]

    x = torch.zeros(1, 1, vocab_size)  # x^{<1>}: the zero vector
    hidden = None
    sampled_ix: list[int] = []
    with torch.no_grad():
        for _ in range(max_length):
            logits, hidden = model(x, hidden)  # logits (1, 1, V)
            probabilities = torch.softmax(logits[0, 0] / temperature, dim=-1)
            index = int(torch.multinomial(probabilities, 1, generator=generator))
            sampled_ix.append(index)
            if index == newline_ix:
                break
            x = torch.zeros(1, 1, vocab_size)
            x[0, 0, index] = 1.0  # feed the drawn character back as x^{<t+1>}
    return "".join(ix_to_char[index] for index in sampled_ix)


def main() -> None:
    """Train the char-LM on the fixture corpus and print the anchors and a sample."""
    fixture = make_fixture()
    model = build_model(fixture["vocab_size"], fixture["hidden_size"])
    examples = fixture["examples"]

    initial = float(corpus_cross_entropy(model, examples).detach())
    print(f"vocab size V:      {fixture['vocab_size']}")
    print(f"ln V:              {math.log(fixture['vocab_size']):.4f}")
    print(f"init cross-entropy: {initial:.4f}  (perplexity {perplexity(initial):.2f})")

    history = train(model, examples, steps=300)
    print(
        f"final cross-entropy: {history[-1]:.4f}  "
        f"(perplexity {perplexity(history[-1]):.2f})"
    )

    generator = torch.Generator().manual_seed(fixture["sample_seed"])
    for temperature in (1.0, 0.5):
        text = sample(
            model,
            fixture["char_to_ix"],
            fixture["ix_to_char"],
            generator=generator,
            temperature=temperature,
        )
        print(f"sample (T={temperature}): {text!r}")


if __name__ == "__main__":
    main()
