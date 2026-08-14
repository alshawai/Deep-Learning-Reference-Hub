# Deep Learning Reference Hub 🧠

<div align="center">

[![CI](https://github.com/alshawai/Deep-Learning-Reference-Hub/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/alshawai/Deep-Learning-Reference-Hub/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-MkDocs-526CFE?logo=materialformkdocs&logoColor=white)](https://alshawai.github.io/Deep-Learning-Reference-Hub/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/pyproject.toml)
[![NumPy 1.26+](https://img.shields.io/badge/NumPy-1.26%2B-013243?logo=numpy&logoColor=white)](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/pyproject.toml)
[![PyTorch 2.3+](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C?logo=pytorch&logoColor=white)](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/pyproject.toml)
[![TensorFlow 2.17+](https://img.shields.io/badge/TensorFlow-2.17%2B-FF6F00?logo=tensorflow&logoColor=white)](https://github.com/alshawai/Deep-Learning-Reference-Hub/blob/main/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Last updated](https://img.shields.io/github/last-commit/alshawai/Deep-Learning-Reference-Hub?label=Last%20updated&color=blue)](https://github.com/alshawai/Deep-Learning-Reference-Hub/commits/main)

</div>

Deep learning is easier to understand when the equations, implementation, and
engineering decisions are visible together. This hub pairs readable,
from-scratch implementations with mathematical explanations, practical guides,
and concise reference material.

**[Read the documentation](https://alshawai.github.io/Deep-Learning-Reference-Hub/)**

## What you will find here

- **Readable implementations** of optimizers, hyperparameter-search methods,
  neural-network building blocks, and training techniques.
- **Mathematical explanations** that derive the algorithms rather than treating
  them as black boxes.
- **Practical how-to guides** for choosing optimizers, checking gradients,
  tuning learning rates, and running hyperparameter searches.
- **Reference pages** for equations, defaults, tensor shapes, and the public
  Python API.
- **Tests grounded in the mathematics**, including hand-computed values,
  closed-form results, and finite-difference checks.

The documentation follows [Diátaxis](https://diataxis.fr/), so learning,
problem-solving, lookup, and deeper understanding each have a clear home:

| If you want to... | Start here |
| --- | --- |
| Learn by building something | [Tutorials](https://alshawai.github.io/Deep-Learning-Reference-Hub/tutorials/) |
| Complete a specific task | [How-to guides](https://alshawai.github.io/Deep-Learning-Reference-Hub/how-to/) |
| Look up an equation, default, shape, or API | [Reference](https://alshawai.github.io/Deep-Learning-Reference-Hub/reference/) |
| Understand why a method works | [Explanation](https://alshawai.github.io/Deep-Learning-Reference-Hub/explanation/) |

## Suggested learning paths

The hub is a reference rather than a fixed course, but these routes provide a
useful order through the material that exists today.

### Foundations

1. [Forward and backward propagation](https://alshawai.github.io/Deep-Learning-Reference-Hub/explanation/forward-and-backward-propagation/)
2. [Network shapes and dimensions](https://alshawai.github.io/Deep-Learning-Reference-Hub/reference/network-shapes-and-dimensions/)
3. [Matrix calculus for neural networks](https://alshawai.github.io/Deep-Learning-Reference-Hub/explanation/matrix-calculus-for-neural-networks/)
4. [Run a gradient check](https://alshawai.github.io/Deep-Learning-Reference-Hub/how-to/run-a-gradient-check/)

### Training a model well

1. [Parameter initialization](https://alshawai.github.io/Deep-Learning-Reference-Hub/explanation/parameter-initialization/)
2. [Regularization](https://alshawai.github.io/Deep-Learning-Reference-Hub/explanation/regularization/)
3. [Optimization algorithms](https://alshawai.github.io/Deep-Learning-Reference-Hub/explanation/optimization-algorithms/)
4. [Choose an optimizer](https://alshawai.github.io/Deep-Learning-Reference-Hub/how-to/choose-an-optimizer/)
5. [Tune a learning rate](https://alshawai.github.io/Deep-Learning-Reference-Hub/how-to/tune-learning-rate/)

### Hyperparameter optimization

1. [The hyperparameter-tuning landscape](https://alshawai.github.io/Deep-Learning-Reference-Hub/explanation/hyperparameter-tuning-landscape/)
2. [Hyperparameter search methods](https://alshawai.github.io/Deep-Learning-Reference-Hub/reference/hyperparameter-search-methods/)
3. [Run a hyperparameter search](https://alshawai.github.io/Deep-Learning-Reference-Hub/how-to/run-hyperparameter-search/)

## Install

Clone the repository and install the NumPy reference package:

```bash
git clone https://github.com/alshawai/Deep-Learning-Reference-Hub.git
cd Deep-Learning-Reference-Hub
python -m pip install -e .
```

The base installation includes NumPy, SciPy, and Matplotlib. NumPy is the
reference platform used by the published implementation modules.

Some documentation also demonstrates the same ideas with PyTorch 2.3+ and
TensorFlow 2.17+ using its bundled Keras API. The framework badges communicate
the intended minimum compatibility for those optional examples. They do not
claim that every NumPy implementation already has a framework port; expanding
that parity is part of the hub's future growth.

Install the optional frameworks when you need those examples:

```bash
python -m pip install -e ".[frameworks]"
```

## Run the gate

Install the development and documentation dependencies:

```bash
python -m pip install -e ".[dev,docs]"
```

Then run the same checks enforced by CI:

```bash
python tools/hubcheck.py all
mkdocs build --strict
ruff format --check .
ruff check .
python -m pytest -q
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the repository conventions and
contribution workflow.

## Navigate by experience and interest

### By depth

- **Starting out:** forward propagation, activation functions, parameter
  initialization, and the optimizer-selection guide.
- **Building confidence:** backpropagation, regularization, gradient checking,
  learning-rate schedules, and random search.
- **Going deeper:** matrix calculus, adaptive optimization, Bayesian
  optimization, multi-fidelity methods, and population-based training.

These labels describe the background a page assumes, not the importance of its
subject. The [documentation navigation](https://alshawai.github.io/Deep-Learning-Reference-Hub/)
is the authoritative index as the hub grows.

### By subject

- **Neural-network foundations:** propagation, activations, tensor shapes, and
  parameter updates.
- **Training techniques:** initialization, regularization, gradient checking,
  early stopping, optimizers, and learning-rate schedules.
- **Hyperparameter optimization:** random and Bayesian search, ASHA and
  multi-fidelity methods, population-based training, and learning-rate finding.

Computer vision, natural language processing, and generative modelling are
natural future directions, but the README does not list them as current coverage
until the repository contains material readers can use.

### By framework

- **NumPy:** the current from-scratch reference implementations.
- **PyTorch:** optional examples targeting PyTorch 2.3 and newer.
- **TensorFlow/Keras:** optional examples targeting TensorFlow 2.17 and newer,
  using the Keras API bundled with TensorFlow.

## Quality standards

The hub aims to be educational without becoming approximate. Contributions are
expected to preserve:

- **Mathematical accuracy:** equations and numerical behavior agree.
- **Readable implementations:** teaching code exposes the algorithm's important
  steps instead of hiding them behind abstractions.
- **Behavioral tests:** tests assert meaningful values and invariants, not only
  output shapes.
- **Reproducibility:** examples state seeds, shapes, dtypes, and tolerances when
  those details affect the result.
- **Documentation integrity:** strict site builds reject broken links, missing
  navigation entries, and invalid cross-references.
- **Consistent Python quality:** Ruff enforces formatting, imports, and NumPy
  docstring conventions.

## Repository statistics

- **Total Documents**: 21
- **Code Examples**: 17 implementations
- **Frameworks Covered**: NumPy

The framework badges above describe optional example compatibility. The checked
statistics count published implementation modules, which are currently NumPy
based.

## Contributing

Corrections, clearer explanations, stronger tests, new implementations, and
carefully chosen references are welcome. Start with
[CONTRIBUTING.md](CONTRIBUTING.md), and use
[GitHub Issues](https://github.com/alshawai/Deep-Learning-Reference-Hub/issues)
to propose or discuss larger changes.

## Acknowledgements

This project builds on the work of researchers, educators, and open-source
maintainers who make deep learning knowledge accessible. In particular, it
owes much to Andrew Ng and the Deep Learning Specialization, the authors of the
papers and books cited throughout the documentation, and the NumPy, SciPy,
Matplotlib, PyTorch, TensorFlow, and Keras communities.

## License

This project is available under the [MIT License](LICENSE).

---

<div align="center">

### Learn the mathematics. Read the implementation. Verify the behavior.

_A living deep learning reference, built to be understood and improved._

</div>
