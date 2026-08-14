// MathJax 3, configured for the math `pymdownx.arithmatex` hands it.
//
// In generic mode arithmatex has already rewritten every `$...$` and `$$...$$`
// into `\(...\)` and `\[...\]` inside a span it marks, so the delimiters below
// are deliberately not the dollar signs the Markdown sources use.
//
// `processEnvironments` is load-bearing rather than decorative: the derivations
// in this hub are written as `\begin{align}` blocks, which is a MathJax
// extension. A renderer without it shows those equations as raw source.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

// Material loads pages without a full reload, so equations on a page arrived at
// by navigation are never typeset unless this runs again on each change.
document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
