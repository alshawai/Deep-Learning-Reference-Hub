# Contributing to Deep Learning Reference Hub

Thank you for considering contributing to this project!  
The **Deep Learning Reference Hub** aims to provide **clean, well-documented, and educational implementations** of core deep learning concepts.

Please follow these guidelines to keep the repository consistent and high-quality.

---

## 📌 Table of Contents

- [Contributing to Deep Learning Reference Hub](#contributing-to-deep-learning-reference-hub)
  - [Table of Contents](#-table-of-contents)
  - [Code of Conduct](#-code-of-conduct)
  - [Getting Started](#️-getting-started)
  - [Coding Standards](#-coding-standards)
    - [Where Code Lives](#where-code-lives)
    - [Running the Gate](#running-the-gate)
    - [Tests](#tests)
  - [Documentation Standards](#-documentation-standards)
    - [Module-Level Docstrings](#module-level-docstrings)
    - [Class and Function Docstrings](#class-and-function-docstrings)
  - [Adding New Resources](#-adding-new-resources)
  - [Pull Requests](#-pull-requests)
    - [Commit Message Guidelines](#commit-message-guidelines)

---

## ✅ Code of Conduct

Be respectful and constructive. Discussions should stay technical and educational.

---

## ⚙️ Getting Started

1. Fork the repository and clone it locally.
2. Install the hub and its development tools:
```bash
pip install -e ".[dev,docs]"
```
This installs `dlhub` in editable mode, so an edit to a module is live in the next test run with no reinstall. Add the frameworks with `pip install -e ".[dev,docs,frameworks]"` if you are working on a port.
3. Create a new branch for your changes:
```bash
git checkout -b feature/my-new-feature
```

--- 

## 📝 Coding Standards

- Follow PEP 8 for Python code style.
- `pyproject.toml` holds the real configuration — line length, lint rules, docstring convention. It is the authority. Where this page and `pyproject.toml` disagree, `pyproject.toml` is right.
- Ruff does both the formatting and the linting. There is no separate formatter or docstring checker to run.

### Where Code Lives

An implementation goes inside the `dlhub` package, under `src/`, in the subpackage for its subject. A reader installs the hub and imports it — `from dlhub.optimizers.adam import AdamOptimizer` — so a module that sits outside the package is one a reader cannot reach.

A new subpackage needs an `__init__.py` exporting the names it means to publish. That file is also how `hubcheck` recognises an implementation, so a module in a directory without one is invisible to the README's count.

### Running the Gate

Run these five commands before you open a pull request. [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs the same checks, so a clean run here means a green CI.

```bash
python tools/hubcheck.py all   # links and anchors outside docs/, prose Python, README claims
mkdocs build --strict          # the documentation site, and every link inside it
ruff format .                  # apply formatting
ruff check .                   # lint, including the NumPy docstring rules
python -m pytest -q            # tests
```

`ruff format .` rewrites files. Read the diff it produces instead of committing it blind. It also reformats the code inside docstring examples, so a formatting pass touches docstrings as well as code.

`hubcheck` checks what a linter cannot: that every relative link resolves, that every anchor points at a heading that exists, that the Python inside fenced code blocks compiles, and that the counts `README.md` advertises match the files actually in the tree. Run one check on its own by passing `links`, `anchors`, `fences`, or `readme` instead of `all`.

`mkdocs build --strict` builds the documentation site under `docs/`, and fails on a broken internal link, a link to a heading that does not exist, or a page missing from the nav in `mkdocs.yml`. Adding a page to `docs/` therefore means adding its nav entry in the same commit. Preview the site with `mkdocs serve`.

The two tools split the tree between them rather than overlapping. `mkdocs build --strict` resolves the links and anchors inside `docs/`; `hubcheck`'s `links` and `anchors` checks cover everything outside it, which today is `README.md`, `CONTRIBUTING.md`, and the documents not yet filed into the site. That boundary is read from `docs_dir` in `mkdocs.yml`, so it follows the tree if the tree moves, and the second half shrinks as documents are filed in. Pointing `hubcheck links --file` at a page inside `docs/` reports that the site build owns it rather than passing on an empty scan. The `fences` and `readme` checks are not split — no site build compiles the Python in a fence or counts what the README claims.

If a command is missing, install the development tools:

```bash
pip install -e ".[dev,docs]"
```

### Tests

A new implementation needs tests. Put them in `tests/`, named after the module they cover.

Test the mathematics, not the plumbing. A test that only checks an output's shape still passes when the sign of the update is backwards. Assert against a value you worked out by hand, a closed-form result, or a finite-difference gradient check.

Name a test after the property it holds the code to, and reserve the docstring for what breaks when it fails. `test_the_population_thins_by_the_reduction_factor_at_each_rung` tells a reader what is being claimed; `"""Test the population."""` tells them nothing, which is why `tests/` is exempt from the docstring-presence rules in `pyproject.toml`.

A framework port needs a parity test: the same fixture through the NumPy reference and through the port, compared within a stated tolerance. Declare the tolerance in the test, and say why it is the number it is. CI runs parity tests in a separate job that installs PyTorch and TensorFlow, since most authoring environments do not have them.

A parity test that skips itself when a dependency is missing is worse than no test at all, because it reports green. Let it fail instead.

--- 

## 📖 Documentation Standards

We use NumPy style docstrings across the repository.  
Each module must include a module-level docstring, and each public function or class should have clear parameter/return documentation.

### Module-Level Docstrings

Use the following format for all module-level docstrings:
```python 
"""
<Module Title>
=========================

A brief but clear description of what this module does, its purpose, and context
in deep learning. Mention if it's an implementation, utility, or theoretical demonstration.

References
----------
- <Author(s)>. <Title of Paper or Book>. <Publisher/Conference>, <Year>.
  <URL if applicable>

Author
------
<Your Name or Team Name> (Deep Learning Reference Hub)

License
-------
MIT License

Notes
-----
Any special considerations, numerical stability warnings, or implementation notes.
"""
```
Example:
```python 
"""
Adam Optimizer
==============

Implements Adaptive Moment Estimation (Adam) for stochastic optimization.

References
----------
- Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.
  https://arxiv.org/abs/1412.6980

Author
------
Deep Learning Reference Hub

License
-------
MIT License
"""
```

### Class and Function Docstrings

Example (NumPy style):
```python 
class AdamOptimizer:
    """
    Adam (Adaptive Moment Estimation) Optimizer.

    Combines the benefits of AdaGrad and RMSProp by computing adaptive learning
    rates using first and second moment estimates.

    Parameters
    ----------
    learning_rate : float, default=0.001
        Step size for parameter updates.
    beta1 : float, default=0.9
        Exponential decay rate for first moment estimates.
    beta2 : float, default=0.999
        Exponential decay rate for second moment estimates.
    """


def update_parameters(params: Dict, grads: Dict, t: int) -> Dict:
    """
    Update model parameters using Adam optimization.

    Parameters
    ----------
    params : dict
        Model parameters to be updated.
    grads : dict
        Gradients for each parameter.
    t : int
        Timestep for bias correction.

    Returns
    -------
    dict
        Updated parameters after applying Adam step.
    """
```

---

## 🔗 Adding New Resources
To add books, papers, or courses:
1. Open `docs/reference/resources.md`.
2. Add the resource under the appropriate section.
3. Use this format:
```markdown
- **Title (Author, Year)** – [link](https://example.com)
```
Only add well-established, high-quality resources.

---

## ✅ Pull Requests

1. Ensure the gate passes locally — see [Running the Gate](#running-the-gate):
```bash
python tools/hubcheck.py all
mkdocs build --strict
ruff format .
ruff check .
python -m pytest -q
```
2. Write clear commit messages.
3. Reference any related issue in your PR description.
4. PRs will be reviewed for:
    - Correctness
    - Code clarity
    - Proper documentation (docstrings, updated resource reference if needed)
    - Tests covering any new implementation

### Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) standard to ensure a clean and meaningful commit history.

- Use the appropriate prefix, such as:
  - `feat:` for new features or implementations
  - `fix:` for bug fixes
  - `docs:` for documentation updates (including docstrings)
  - `test:` for adding or updating tests
  - `chore:` for maintenance, dependency updates, or repo structure changes

Example:
```
feat: add RMSprop optimizer with full documentation
```
Keeping a consistent commit style helps maintainers review PRs efficiently and improves changelog generation.


**Thank you for helping improve this project! 🚀**
