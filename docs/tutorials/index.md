# Tutorials

A tutorial is a lesson. It takes a reader who does not yet know the subject and
walks them through building one working thing, end to end, making every decision
for them along the way.

## Tutorials here

- **[Recurrent neural networks: watching gradients vanish through time](01-recurrent-neural-networks.ipynb)**
  — a runnable notebook. You supervise a single timestep of a vanilla RNN,
  back-propagate through time, and measure how much of the learning signal
  survives the trip back to each earlier step, turning the vanishing-gradient
  problem from a claim into a curve you have plotted. It imports the hub's
  canonical `dlhub.nn.sequence.rnn`, so the lesson cannot drift from the code it
  teaches.

- **[LSTM and GRU: the gate that decides what a gradient survives](02-lstm-and-gru.ipynb)**
  — a runnable notebook, and the answer to the problem the first one measures.
  You difference the imported LSTM cell to find the forget gate sitting on the
  diagonal of $\partial c^{\langle t\rangle} / \partial c^{\langle t-1\rangle}$,
  watch five units of one cell choose five different memory timescales, then pin
  that gate by hand and measure what it does to the gradient reaching the start
  of the sequence. It imports `dlhub.nn.sequence.lstm`, `.gru`, and
  `.min_gated`, and implements nothing of its own.

- **[Word embeddings: from one-hot to a vector you can do arithmetic on](04-word-embeddings.ipynb)**
  — a runnable notebook. You start from a one-hot column, replace it with a
  lookup into a small planted embedding matrix, and then *use* the vectors:
  measure meaning with cosine, run `king − man + woman` through a 3CosAdd
  analogy, rank nearest neighbours, and project the whole vocabulary to two
  dimensions with PCA to see the geometry the arithmetic relies on. It imports
  the hub's canonical `dlhub.embeddings`, so the lesson cannot drift from the
  code it teaches.

## Framework tutorial notebooks

The three notebooks listed before this heading import only what the docs build
installs — numpy, scipy, and the `dlhub` package — so `mkdocs build --strict`
re-executes them and their committed outputs cannot drift from the code that
produced them. A tutorial that imports a framework instead — PyTorch or
TensorFlow — cannot run in that framework-free build. Such a notebook is a
*framework tutorial notebook*, and it carries three obligations:

- Its first cell is a banner that names the framework and gives the install
  command, `pip install -e '.[frameworks]'`, so a reader knows what the lesson
  needs before the first import runs.
- It is listed under `execute_ignore` in `mkdocs.yml`, so the docs build renders
  it from its committed outputs rather than executing it.
- The `framework-notebooks` job in `.github/workflows/ci.yml` installs the
  framework and executes it end to end, so a broken cell or a stale output still
  fails CI.

The hub's framework tutorial notebooks:

- **[Language modeling and sampling: train a character model, then turn the temperature knob](03-language-modeling-and-sampling.ipynb)**
  — a runnable PyTorch notebook. You train a character-level RNN language model,
  watch its perplexity fall from uniform guessing among 18 characters to fewer
  than four, sample novel words from it, and slide the temperature between
  cautious and reckless without touching a weight. It imports the canonical
  `dlhub.pytorch.sequence.language_model`.

## Still missing

**A from-scratch construction of an L-layer network**: initialise the
parameters, implement one forward pass, derive and implement one backward pass,
verify the gradients numerically, then train it on a small problem the reader
can watch converge. The
[forward- and backward-propagation derivation](../explanation/forward-and-backward-propagation.md)
it draws on is already written; what is missing is the guided path through it.

Writing that tutorial is a content project rather than a documentation move, so
it remains out of scope here — recorded as a gap rather than hidden.

## What belongs here

A page belongs in this section when a beginner following it in order arrives at
something that works.

- It has a single, stated destination, and reaching it is the point.
- It is safe to follow without judgment: every choice is made for the reader,
  and none of them are presented as options.
- It is complete. A tutorial that leaves a reader with a broken artifact has
  failed even if every individual step was correct.
- It shows results at each step, so a reader can tell they are still on track.

## What does not belong here

- **A task with a goal the reader already has.** That is a how-to guide. The
  distinction is who chose the destination: in a tutorial the author did, in a
  how-to the reader did.
- **The reasoning behind a design.** That is an explanation. A tutorial may say
  "use He initialisation here"; it should not stop to derive why the variance
  scales that way.
- **A catalogue of options, parameters, or defaults.** That is reference. A
  tutorial names the one value it wants the reader to type.
- **A lesson that assumes prior familiarity with its own subject.** That is not
  a tutorial at all, and it is usually a how-to guide in the wrong section.
