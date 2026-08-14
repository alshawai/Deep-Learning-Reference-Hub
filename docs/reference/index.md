# Reference

Reference describes the machinery. A reader arrives knowing what they want and
needing a precise detail: an equation, a shape, a default, a signature.

Reference is consulted, not read. Its organisation should mirror the thing it
describes rather than any path through it.

## What belongs here

- **Equations, stated rather than derived.** The update rule as it is applied.
  The derivation that produces it is an explanation.
- **Tables**: shapes and dimensions, default values, comparisons between methods
  on stated criteria.
- **API documentation**, generated from the docstrings so that the published
  reference and the installed package cannot disagree.
- Anything a reader would return to repeatedly for one line at a time.

Accuracy is the only virtue that matters here. A reference page is austere on
purpose: it makes no argument, teaches nothing, and takes no position on what a
reader should do.

## What does not belong here

- **Why a default is what it is.** That is an explanation. Reference states that
  `beta2` defaults to `0.999`; the reason that value works belongs elsewhere.
- **A recommendation.** "Use Adam unless you have a reason not to" is a how-to
  or an explanation. Reference lists what each optimizer does and lets the
  reader choose.
- **A worked example that teaches.** A short illustrative snippet is fine; a
  guided sequence is a tutorial.
- **Hand-written API documentation.** The API pages are generated. A hand-copied
  signature is the first thing in a repository to go stale, and the docstring is
  already required by the house style.

## Planned contents

Drawn from the lookup material currently embedded in the source documents:

- **[Network shapes and dimensions](network-shapes-and-dimensions.md)** — the forward and backward dimension tables
  from the L-layer derivation, and its dimensional-analysis verifications.
- **[Activation functions](activation-functions.md)** — each function with its derivative.
- **[Parameter update rule](parameter-update-rule.md)** — the gradient-descent
  update equations for an L-layer network.
- **[Optimizer update rules](optimizer-update-rules.md)** — the update equation and default hyperparameters
  for each method, plus the comparison table.
- **[Learning rate schedules](learning-rate-schedules.md)** — each schedule's formula and parameters.
- **[Initialisation schemes](parameter-initialization.md)** — the variance each scheme uses and when it applies.
- **[Regularisation methods](regularization.md)** — equations, implementations,
  and framework examples for regularisation techniques.
- **[Gradient checking](gradient-checking.md)** — result tolerances and links to
  the generated implementation reference.
- **[Hyperparameter search methods](hyperparameter-search-methods.md)** — the method comparison table and the
  documented defaults.
- **[Deep learning resources](resources.md)** — curated books, papers, courses,
  frameworks, and supplementary reading.

## API reference

Generated from the package docstrings, one page per subpackage:

- [Optimizers](api/optimizers.md) — gradient descent and the adaptive methods.
- [Tuning](api/tuning.md) — hyperparameter search strategies.
- [Neural networks](api/nn.md) — network construction and the training loop.
- [Training techniques](api/training.md) — early stopping and gradient checking.
