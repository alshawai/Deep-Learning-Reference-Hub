# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses **editorial versioning**: a *minor* release adds a completed
topic, a *major* release marks a project milestone, and a *patch* corrects
existing material. The version records the hub's growth, not a promise of Python
API stability — see the versioning policy in
[`CONTRIBUTING.md`](CONTRIBUTING.md#versioning-and-releases).

## [Unreleased]

## [1.5.1] - 2026-10-08

### Fixed
- The reference, how-to, and explanation indexes still called their sections
  "Planned contents" and omitted eleven pages that had already shipped, so a
  reader browsing them would conclude the sequence-model material did not exist.
  The sections are now "Contents" and list every published page.
- Six crosswalk sections carried a preamble saying sibling pages "are not yet
  published" when every row beneath them was already a live link.
- Nine internal links and one clone command named the repository owner
  `eima40x4c`, which resolves only through a GitHub redirect; they now name
  `alshawai`, matching the remote and the rest of the documentation.
- `release.yml` and `tools/releasecheck.py` pointed at
  `docs/explanation/versioning-and-releases.md`, a page that does not exist.
  Both now cite the versioning policy where it actually lives, in
  `CONTRIBUTING.md`.
- The recurrent-neural-networks crosswalk sent readers to a topic slug as a
  placeholder for a how-to guide that has since shipped.
- The tutorials index described its framework-free notebooks as "the three
  notebooks above" while a fourth sat below the heading in another section.
- The `README.md` subject list omitted modern sequence models, and no
  sequence-model page appeared in any of the three depth tiers.

### Removed
- `.pylintrc`, a UTF-16 encoded configuration file that no parser could read and
  no part of the toolchain referenced.

## [1.5.0] - 2026-10-08

### Added
- [Word embeddings](docs/explanation/word-embeddings.md): static word vectors
  and their geometry — the embedding matrix and lookup identity, cosine
  similarity, nearest neighbours, analogy arithmetic, and PCA projection — with
  an explanation, a reference, a from-scratch NumPy implementation, a PyTorch
  `nn.Embedding` idiom port, and a tutorial notebook.
- A tag-triggered GitHub Release workflow, and a `releasecheck` CI guard that
  keeps the version, the latest tag, and this file in agreement.
- The versioning and release policy, documented in `CONTRIBUTING.md`.

### Changed
- The package version is single-sourced from `dlhub.__version__`.
- Tidied the shared internals of the `dlhub.nn.sequence` package.

## [1.4.0] - 2026-09-30

### Added
- [Modern sequence models](docs/explanation/modern-sequence-models.md): the
  topic covering the architectures that followed the recurrent cell —
  explanation, reference, and implementation.

## [1.3.0] - 2026-09-27

### Added
- [Language modeling and sampling](docs/explanation/language-modeling-and-sampling.md):
  explanation, reference, a how-to guide for training a language model, and a
  tutorial notebook.
- A framework-notebook toolchain, so a tutorial notebook may import a framework
  and still be executed in CI.

## [1.2.0] - 2026-09-26

### Added
- [LSTM and GRU](docs/explanation/lstm-and-gru.md): the gated-cell topic —
  explanation, reference, a from-scratch NumPy implementation, a tutorial
  notebook, and TensorFlow parity.

## [1.1.0] - 2026-09-12

### Added
- [Recurrent neural networks](docs/explanation/recurrent-neural-networks.md):
  the first sequence topic — explanation, reference, a from-scratch NumPy
  implementation, a PyTorch port, and a tutorial notebook.
- An executable-notebook toolchain, so the tutorials run end to end in CI.

### Fixed
- ASHA now reports an empty run rather than raising, records the evaluations in
  flight when a run ends, and preserves its concurrency invariant.

## [1.0.0] - 2026-08-14

Initial public release: the point the hub became a tested, documented, published
package rather than a folder of notes.

### Added
- The `dlhub` package (`src/` layout) with the `optimizers`, `tuning`, `nn`, and
  `training` subpackages.
- A documentation site in the [Diátaxis](https://diataxis.fr/) structure
  (tutorials, how-to guides, reference, explanation), published to GitHub Pages,
  with an API reference generated from the docstrings.
- Optimization: mini-batch gradient descent, Momentum, RMSprop, Adam,
  exponential weighted averages, learning-rate schedules, and a comparison
  harness.
- Hyperparameter tuning: random search, Bayesian optimization, a learning-rate
  finder, ASHA multi-fidelity optimization, and population-based training.
- Core network material: the L-layer fully-connected network, parameter
  initialization, regularization, and gradient checking.
- A CI quality gate running ruff, pytest, the `hubcheck` integrity checks, and a
  strict documentation build.

[Unreleased]: https://github.com/alshawai/Deep-Learning-Reference-Hub/compare/v1.5.1...HEAD
[1.5.1]: https://github.com/alshawai/Deep-Learning-Reference-Hub/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/alshawai/Deep-Learning-Reference-Hub/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/alshawai/Deep-Learning-Reference-Hub/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/alshawai/Deep-Learning-Reference-Hub/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/alshawai/Deep-Learning-Reference-Hub/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/alshawai/Deep-Learning-Reference-Hub/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/alshawai/Deep-Learning-Reference-Hub/releases/tag/v1.0.0
