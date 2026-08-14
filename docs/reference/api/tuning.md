# Tuning

<!-- hubcheck: generated -->

Search strategies for choosing hyperparameters, from random sampling through
model-based search to the multi-fidelity methods that spend their budget unevenly
on purpose.

One entry point per method, plus a dispatcher that takes a method name. The
per-method functions take that method's own search-space format and return its
own result type; the dispatcher takes the framework's and returns an
`ExperimentResult`.

::: dlhub.tuning
    options:
      show_submodules: true
