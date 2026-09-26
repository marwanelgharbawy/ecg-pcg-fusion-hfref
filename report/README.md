# CardioFusion report

Write everything in `main.tex`. Comments beginning with `% -------------`
separate the sections. Find the ECG, PCG, or fusion heading and edit there.

On Overleaf, upload `main.tex` and any figures, then click Recompile.

Using VS Code Shortcuts:
- **Ctrl+Alt+B to build**
- **Ctrl+Alt+V to preview** the PDF in a tab.

Saving the file also rebuilds it. If an old failed preview is open, close
that tab and reopen the preview after the build finishes.

The workspace settings use the installed MiKTeX compiler directly and run
it twice to resolve references. This report does not need Perl or BibTeX.
The compiler path in `.vscode/settings.json` is specific to this computer;
change it if using another computer. Overleaf does not use that settings file.
