#!/usr/bin/env sh
# Build main.pdf. Uses latexmk if available; ENGINE=pdflatex (default) or lualatex.
set -e
cd "$(dirname "$0")"
ENGINE="${ENGINE:-pdflatex}"
$ENGINE -interaction=nonstopmode -halt-on-error main.tex >/dev/null
bibtex main >/dev/null
$ENGINE -interaction=nonstopmode -halt-on-error main.tex >/dev/null
$ENGINE -interaction=nonstopmode -halt-on-error main.tex | grep -E "Output written|Overfull" || true
