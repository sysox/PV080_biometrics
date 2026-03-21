#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# Config
# ----------------------------
PRINTER="${PRINTER:-M2400}"

# ----------------------------
# Args
# ----------------------------
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <file.(pdf|docx|md)>"
  exit 1
fi

INPUT="$1"

# ----------------------------
# Helpers
# ----------------------------
info() { echo "[INFO] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

need() {
  command -v "$1" >/dev/null 2>&1 || die "Missing dependency: $1"
}

# ----------------------------
# Checks
# ----------------------------
[[ -f "$INPUT" ]] || die "File not found: $INPUT"

need lp
need pdftops

EXT="${INPUT##*.}"
PDF=""

# ----------------------------
# Convert → PDF
# ----------------------------
case "$EXT" in
  pdf)
    PDF="$INPUT"
    ;;
  docx)
    need libreoffice
    info "DOCX → PDF"
    libreoffice --headless --convert-to pdf "$INPUT" --outdir .
    PDF="${INPUT%.docx}.pdf"
    ;;
  md)
    need pandoc
    info "MD → PDF"
    PDF="$(basename "${INPUT%.md}.pdf")"
    pandoc "$INPUT" -o "$PDF"
    ;;
  *)
    die "Unsupported format: .$EXT"
    ;;
esac

# ----------------------------
# PDF → PS (critical for M2400)
# ----------------------------
PS="$(basename "${PDF%.pdf}.ps")"
info "PDF → PS"
pdftops "$PDF" "$PS"

# ----------------------------
# Print
# ----------------------------
info "Printing on $PRINTER"
lp -d "$PRINTER" "$PS"

info "Done."