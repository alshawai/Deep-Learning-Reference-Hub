// MathJax 3, configured for two sources of math that reach the page differently.
//
// On native Markdown pages, `pymdownx.arithmatex` (generic mode) has already
// rewritten every `$...$` and `$$...$$` into `\(...\)` and `\[...\]` inside a
// span it marks `arithmatex` -- so those delimiters, and that class, are what
// MathJax looks for there.
//
// Notebook pages are different: `mkdocs-jupyter` converts each cell with
// nbconvert, which runs *outside* the Markdown pipeline, so arithmatex never
// sees the notebook's Markdown cells and their `$...$`/`$$...$$` survive as raw
// dollar signs. MathJax must therefore also recognise the dollar delimiters, and
// must not be gated to only the `arithmatex` class. This is safe because every
// literal `$` anywhere on the built site is intended math -- the sources use no
// prose or currency dollar signs -- so a `$` delimiter has nothing to misfire on.
//
// The dollar signs we must NOT typeset are the ones inside notebook code: the
// `$...$` matplotlib mathtext in cell inputs and any `$` in printed cell output.
// Those live in `jp-CodeMirrorEditor` (rendered as spans, so `skipHtmlTags`'
// `code`/`pre` does not cover them) and `jp-OutputArea`; ignoring those two
// classes leaves them literal while notebook prose in `jp-RenderedMarkdown`, and
// all native-page content, is still processed.
//
// `processEnvironments` is load-bearing rather than decorative: the derivations
// in this hub are written as `\begin{align}` blocks, which is a MathJax
// extension. A renderer without it shows those equations as raw source.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: "jp-CodeMirrorEditor|jp-OutputArea",
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
