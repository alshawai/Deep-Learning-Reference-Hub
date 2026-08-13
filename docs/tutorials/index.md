# Tutorials

A tutorial is a lesson. It takes a reader who does not yet know the subject and
walks them through building one working thing, end to end, making every decision
for them along the way.

## This section is empty

That is a gap, not an oversight, and it is recorded here rather than hidden.
Nothing in this hub currently takes a reader from nothing to a working network
in a single guided pass — the material that exists explains the mathematics and
documents the implementations, both of which assume a reader who already knows
what they are building.

**The first tutorial should be the construction of an L-layer network from
scratch**: initialise the parameters, implement one forward pass, derive and
implement one backward pass, verify the gradients numerically, then train it on
a small problem the reader can watch converge. The derivation it draws on is
already written; what is missing is the guided path through it.

Writing that tutorial is a content project rather than a documentation move, so
it is deliberately out of scope for the restructuring that created this section.

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
