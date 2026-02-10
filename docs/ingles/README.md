# IEEE Report - English Version

This folder contains the English translation of the IEEE technical report.

## File Structure

- `INFORME_IEEE_EN.tex` - Main LaTeX document (English)
- `compile.sh` - Compilation script
- `README.md` - This file

## How to Compile

### Option 1: Using the script (Recommended)

```bash
cd docs/ingles/
./compile.sh
```

The script will:
1. Run pdflatex 3 times to resolve all citations and references
2. Generate `INFORME_IEEE_EN.pdf`
3. Clean up auxiliary files

### Option 2: Manual compilation

```bash
cd docs/ingles/
pdflatex INFORME_IEEE_EN.tex
pdflatex INFORME_IEEE_EN.tex
pdflatex INFORME_IEEE_EN.tex
```

**Note:** LaTeX requires **3 compilation passes** to properly resolve:
- Pass 1: Generate auxiliary files
- Pass 2: Resolve citations and cross-references
- Pass 3: Final formatting adjustments

## Images

All images are referenced from the `../report/` directory:
- Graphics: `../report/graficas/`
- Test images: `../report/pruebas/`

## Citations

Citations appear as `[?]` after the first compilation. They will be properly resolved after the second and third passes.

## Requirements

- LaTeX distribution (TeX Live, MiKTeX, etc.)
- Required packages: IEEEtran, graphicx, subfigure, float, hyperref, listings

## Output

After successful compilation, you'll get:
- `INFORME_IEEE_EN.pdf` - Final PDF document in English
