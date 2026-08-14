# Explanation

Explanation is the discussion that deepens understanding. A reader arrives
wanting to know *why* — why a method works, why it was designed this way, why
one approach is preferred over another.

Explanation is read rather than consulted, and it is the one section where the
prose may take a position and argue for it.

## What belongs here

- **Derivations.** The chain rule applied through a network, worked out in full.
- **The reasoning behind a design**: why bias correction matters early in
  training, why initialisation variance depends on fan-in.
- **Comparisons that argue**, as opposed to tabulate: what momentum buys over
  plain gradient descent, and what it costs.
- **Context and history**: what problem a method was introduced to solve, and
  what it replaced.
- Connections between topics that no single implementation owns.

An explanation is free to be discursive. It is the right place for a caveat, an
alternative view, or an admission that a practice is convention rather than
consequence.

## What does not belong here

- **Instructions.** The moment a page tells a reader what to type to achieve
  their own goal, it is a how-to guide.
- **A lesson for a newcomer.** That is a tutorial. An explanation may assume a
  reader who already knows the vocabulary.
- **The definitive table of defaults.** That is reference. An explanation may
  discuss why `0.9` is a reasonable momentum coefficient without becoming the
  page a reader consults to find that number.
- **Restated API documentation.** Link to the generated reference instead.

## Planned contents

Drawn from the conceptual material currently embedded in the source documents:

- **[Forward and backward propagation](forward-and-backward-propagation.md)** — the full derivation from the L-layer
  document, including the recursive formula for hidden layers.
- **[Matrix calculus for neural networks](matrix-calculus-for-neural-networks.md)** — the chain rule in matrix form, and
  why vectorisation changes the complexity.
- **[The optimization landscape](optimization-algorithms.md)** — why optimization is difficult, and what
  gradient descent variants trade against each other.
- **[Exponential weighted averages](optimization-algorithms.md#exponential-weighted-averages)** — the shared mechanism underneath momentum,
  RMSprop, and Adam, and why bias correction is needed.
- **[Adaptive methods](optimization-algorithms.md#what-the-methods-trade)** — what per-parameter learning rates buy, and their costs.
- **[Why initialisation matters](parameter-initialization.md)** — vanishing and exploding gradients, symmetry
  breaking, and the variance argument behind each scheme.
- **[Regularisation theory](regularization.md)** — the bias-variance tradeoff, and how L1, L2,
  dropout, and batch normalisation each act on it.
- **[How gradient checking works](gradient-checking.md)** — the finite-difference approximation and why
  it is a debugging tool rather than a training one.
