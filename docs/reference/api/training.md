# Training techniques

Techniques that wrap a training run rather than perform it: stopping it at the
right moment, and verifying that the gradients driving it are correct.

`early_stopping` names both a module and the one function inside it, so the
function is reached through the module path rather than the package root.

::: dlhub.training
    options:
      show_submodules: true
