# How-to guides

A how-to guide gets a competent reader from a goal they already have to a result.
It assumes they know what they are trying to do and answers only *how*.

## What belongs here

A page belongs in this section when its title could begin with "How to" and a
reader would arrive already wanting the outcome.

- It starts from a real goal, stated in the reader's terms rather than the
  implementation's.
- It gives a sequence that works, and it is honest about the choices along the
  way — unlike a tutorial, a how-to may say "if your problem is sparse, do this
  instead," because its reader can judge.
- It stops when the goal is met. Background that does not change what the reader
  types belongs in an explanation.
- It links to reference for the parameters it mentions rather than restating
  them, so a default has exactly one home.

## What does not belong here

- **Teaching the subject from nothing.** That is a tutorial. A how-to may assume
  its reader knows what a learning rate is.
- **Why the method works.** That is an explanation. A guide to running a
  gradient check says what tolerance to compare against; the derivation of the
  finite-difference approximation belongs elsewhere.
- **The full parameter list of a function.** That is reference, and it is
  generated from the docstrings, so a guide that copies it will go stale.
- **A page organised around a component rather than a goal.** "The Adam
  optimizer" is a reference or explanation page. "How to choose an optimizer" is
  a how-to.

## Contents

- **[How to choose an optimizer](choose-an-optimizer.md)** — from the practical guidelines and optimizer
  selection sections of the optimization algorithms document.
- **[How to run a gradient check](run-a-gradient-check.md)** — from the gradient checking procedure,
  including how to read the resulting relative error.
- **[How to tune a learning rate](tune-learning-rate.md)** — from the learning rate finder material and
  the tuning starting strategy.
- **[How to run a hyperparameter search](run-hyperparameter-search.md)** — from the search strategy guidance,
  covering search space design and the pitfalls documented alongside it.
- **[How to choose a gated cell](choose-a-gated-cell.md)** — picking between the LSTM, the GRU, and
  the minimal gated recurrences when a gate is what the task needs.
- **[How to choose a sequence model](choose-a-sequence-model.md)** — choosing between a recurrent
  network, a state-space model, and attention, from the sequence's length and what it costs to process.
- **[How to train a language model and generate text](train-a-language-model.md)** — the training
  loop, what perplexity tells you, and how to sample from the trained distribution.
