# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses **editorial versioning**: a *minor* release adds a completed
topic, a *major* release marks a project milestone, and a *patch* corrects
existing material. The version records the hub's growth, not a promise of Python
API stability — see the versioning policy in
[`CONTRIBUTING.md`](CONTRIBUTING.md#versioning-and-releases).

## [Unreleased]

## [1.8.0] - 2026-10-26

### Added
- [Contextual embeddings](docs/explanation/contextual-embeddings.md): the modern
  overlay that closes the Word Embeddings family — what changes when a word's
  vector depends on the sentence it sits in. It runs from the static ceiling
  (one vector per type) through ELMo and BERT (one vector per token occurrence),
  treats "the embedding" as a pipeline of encoder, pooling (CLS, mean, last
  token, learned head), instruction prefix, and normalisation, then follows
  SBERT and SimCSE into practical sentence vectors, the bi-encoder /
  cross-encoder / late-interaction (ColBERT) retrieval regimes, MTEB and its
  per-task spread, the modern families (E5, BGE, Instructor, Matryoshka), and
  RAG as the pattern built on top. Every performance claim is cited and dated
  rather than asserted. Prose-only by design: ships an explanation, a
  reference, a how-to for choosing an embedding model, and fourteen new
  resources. Transformer and attention internals are deferred to
  [modern sequence models](docs/explanation/modern-sequence-models.md) and the
  future Transformer topics.

### Changed
- The three sibling Word Embeddings pages and the debiasing notebook now link
  the contextual-embeddings topic where they previously called it planned.
- The modern-sequence-models explanation now uses American spelling throughout
  and drops three stale asides.

## [1.7.0] - 2026-10-17

### Added
- [Debiasing word embeddings](docs/explanation/debiasing-word-embeddings.md): the
  geometry of gender bias in a static embedding and why removing it is hard —
  Bolukbasi et al.'s hard debiasing (identify the gender subspace as the top
  principal component of definitional-pair differences, `neutralize` by
  projecting out the `g` component and renormalizing, `equalize` to make an
  equality set symmetric about `g`), with the WEAT association `s`, test
  statistic, effect size `d` (population std, `ddof = 0`), and an exact
  enumerated permutation `p` over `C(6, 3) = 20` partitions — used to
  quantify bias before and after — and the Gonen & Goldberg "Lipstick on a
  Pig" critique, demonstrated: the direct bias (projection on `g` and WEAT
  `d`) drops to zero after `neutralize`, yet the indirect bias survives, with
  the same neutral-neighbour map before and after and a PCA scatter of the
  debiased matrix still showing the gendered occupation clusters. Ships an
  explanation, a reference, a tutorial notebook, a from-scratch NumPy
  implementation with 32 known-answer tests, and three new resources
  (Bolukbasi et al. 2016, Caliskan et al. 2017, Gonen & Goldberg 2019). Soft
  debiasing, learning-time debiasing (GN-GloVe, Zhao et al. 2018), and
  multiclass / racial bias (Manzini et al. 2019) are named in the explanation
  as "where this goes," not implemented.

## [1.6.0] - 2026-10-12

### Added
- [Word2Vec and GloVe](docs/explanation/word2vec-and-glove.md): how static word
  vectors are learned from scratch — skip-gram with negative sampling (SGNS),
  CBOW, and GloVe's weighted least-squares objective on a co-occurrence matrix,
  each loss paired with its exact gradient and a smoothed-unigram negative
  sampler — with an explanation, a reference, a how-to for training word
  vectors, a from-scratch NumPy implementation, a TensorFlow `nce_loss` parity
  port, and a tutorial notebook. The explanation derives why SGNS implicitly
  factorizes a shifted-PMI matrix (Levy & Goldberg) and names fastText's
  subword extension.

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

[Unreleased]: https://github.com/alshawai/Deep-Learning-Reference-Hub/compare/v1.8.0...HEAD
[1.8.0]: https://github.com/alshawai/Deep-Learning-Reference-Hub/compare/v1.7.0...v1.8.0
[1.7.0]: https://github.com/alshawai/Deep-Learning-Reference-Hub/compare/v1.6.0...v1.7.0
[1.6.0]: https://github.com/alshawai/Deep-Learning-Reference-Hub/compare/v1.5.1...v1.6.0
[1.5.1]: https://github.com/alshawai/Deep-Learning-Reference-Hub/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/alshawai/Deep-Learning-Reference-Hub/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/alshawai/Deep-Learning-Reference-Hub/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/alshawai/Deep-Learning-Reference-Hub/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/alshawai/Deep-Learning-Reference-Hub/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/alshawai/Deep-Learning-Reference-Hub/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/alshawai/Deep-Learning-Reference-Hub/releases/tag/v1.0.0
