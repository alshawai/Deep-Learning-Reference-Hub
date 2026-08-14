# Deep Learning Reference Hub

From-scratch implementations of core deep learning methods, written to be read.

Every module in this hub exposes the mathematics rather than hiding it. The
implementations are teaching artifacts first and working code second, though
they are held to both standards: each is covered by tests that assert against
hand-computed values, closed-form results, or finite-difference gradient checks.

## Install

```bash
git clone https://github.com/eima40x4c/Deep-Learning-Reference-Hub.git
cd Deep-Learning-Reference-Hub
pip install -e .
```

That covers NumPy, SciPy, and Matplotlib, which is everything the from-scratch
implementations use. Once installed, a module is reachable the way any package
is:

```python
from dlhub.optimizers.adam import AdamOptimizer
```

Some documents also show the same idea in TensorFlow or PyTorch. To run those,
add the frameworks:

```bash
pip install -e ".[frameworks]"
```

## How this documentation is organised

The four sections answer four different questions, and each page belongs to
exactly one of them. This is the [Diátaxis](https://diataxis.fr/) split, adopted
because a single page that tries to teach a concept, list its defaults, and walk
through a task serves none of the three well.

| Section | Answers | Read it when |
|---|---|---|
| Tutorials | "Can you teach me to build one?" | you are learning by doing, start to finish |
| How-to guides | "How do I accomplish this task?" | you have a goal and need the steps |
| Reference | "What are the equations, shapes, and defaults?" | you know what you want and need the detail |
| Explanation | "Why does this work?" | you want the derivation and the reasoning |

Each section's own page states what belongs there and what does not, so a
contributor can file a new page without reading this one.
