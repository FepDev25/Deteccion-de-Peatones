#!/bin/bash

# Script to compile the IEEE report in English
# Run this from the docs/ingles/ directory

echo "=== Compiling IEEE Report (English) ==="
echo ""

# First pass - generates auxiliary files
echo "Pass 1/3: Generating auxiliary files..."
pdflatex -interaction=nonstopmode INFORME_IEEE_EN.tex

# Second pass - resolves citations and references
echo ""
echo "Pass 2/3: Resolving citations and references..."
pdflatex -interaction=nonstopmode INFORME_IEEE_EN.tex

# Third pass - final compilation for proper formatting
echo ""
echo "Pass 3/3: Final compilation..."
pdflatex -interaction=nonstopmode INFORME_IEEE_EN.tex

echo ""
echo "=== Compilation complete! ==="
echo "Output file: INFORME_IEEE_EN.pdf"
echo ""
echo "Cleaning up auxiliary files..."
rm -f *.aux *.log *.out *.toc *.lof *.lot 2>/dev/null

echo "Done!"
