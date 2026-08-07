#!/usr/bin/env bash
# Genera un zip de paper/ listo para subir a Overleaf
# (New Project → Upload Project). Excluye artefactos de
# compilación y archivos de sistema.
set -euo pipefail

cd "$(dirname "$0")"

OUT_DIR="../overleaf_uploads"
mkdir -p "$OUT_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_FILE="$OUT_DIR/paper_overleaf_${STAMP}.zip"

zip -r "$OUT_FILE" . \
  -x ".DS_Store" \
  -x "*/.DS_Store" \
  -x "make_zip.sh" \
  -x "*.aux" -x "*.log" -x "*.out" -x "*.toc" \
  -x "*.bbl" -x "*.bcf" -x "*.blg" -x "*.run.xml" \
  -x "*.fls" -x "*.fdb_latexmk" -x "*.synctex.gz"

echo "Zip creado en: $OUT_FILE"
