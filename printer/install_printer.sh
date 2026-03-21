cat > /home/x232886/PycharmProjects/PV080_biometrics/printer/print/print_pv080.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------
# Epson AcuLaser M2400 print helper
# Usage:
#   ./print_pv080.sh /path/to/file.pdf
#   ./print_pv080.sh /path/to/file.docx
#   ./print_pv080.sh /path/to/file.md
#
# Optional:
#   PRINTER=Epson-M2400 ./print_pv080.sh file.pdf
# ---------------------------------

DEFAULT_PRINTER_NAME="Epson-M2400"

info() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

need() {
  command -v "$1" >/dev/null 2>&1 || die "Missing dependency: $1"
}

usage() {
  cat <<EOF2
Usage: $0 <filepath.(pdf|docx|md)>

Examples:
  $0 test_print.pdf
  $0 ./docs/pv080_biometrics_overview.md
  $0 /home/user/file.docx

Optional environment variable:
  PRINTER=Epson-M2400
EOF2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

INPUT="$1"
[[ -f "$INPUT" ]] || die "File not found: $INPUT"

need lp
need lpstat
need pdftops
need readlink

SYSTEM_DEFAULT_PRINTER="$(lpstat -d 2>/dev/null | awk -F': ' '/system default destination/ {print $2}' || true)"
REQUESTED_PRINTER="${PRINTER:-}"

if [[ -n "$REQUESTED_PRINTER" ]] && lpstat -p "$REQUESTED_PRINTER" >/dev/null 2>&1; then
  PRINTER="$REQUESTED_PRINTER"
elif [[ -n "$SYSTEM_DEFAULT_PRINTER" ]] && lpstat -p "$SYSTEM_DEFAULT_PRINTER" >/dev/null 2>&1; then
  PRINTER="$SYSTEM_DEFAULT_PRINTER"
elif lpstat -p "$DEFAULT_PRINTER_NAME" >/dev/null 2>&1; then
  PRINTER="$DEFAULT_PRINTER_NAME"
else
  echo "[ERROR] No usable printer queue found."
  echo
  echo "Available printers:"
  lpstat -p 2>/dev/null || true
  exit 1
fi

INPUT_ABS="$(readlink -f "$INPUT")"
INPUT_DIR="$(dirname "$INPUT_ABS")"
INPUT_FILE="$(basename "$INPUT_ABS")"
INPUT_STEM="${INPUT_FILE%.*}"
EXT="${INPUT_FILE##*.}"
EXT="$(printf '%s' "$EXT" | tr '[:upper:]' '[:lower:]')"

PDF_PATH=""
PS_PATH="$INPUT_DIR/$INPUT_STEM.ps"

case "$EXT" in
  pdf)
    PDF_PATH="$INPUT_ABS"
    ;;
  docx)
    need libreoffice
    info "DOCX -> PDF"
    libreoffice --headless --convert-to pdf "$INPUT_ABS" --outdir "$INPUT_DIR" >/dev/null 2>&1
    PDF_PATH="$INPUT_DIR/$INPUT_STEM.pdf"
    [[ -f "$PDF_PATH" ]] || die "Failed to create PDF from DOCX."
    ;;
  md)
    need pandoc
    info "MD -> PDF"
    PDF_PATH="$INPUT_DIR/$INPUT_STEM.pdf"
    pandoc "$INPUT_ABS" -o "$PDF_PATH"
    [[ -f "$PDF_PATH" ]] || die "Failed to create PDF from Markdown."
    ;;
  *)
    die "Unsupported format: .$EXT (supported: .pdf, .docx, .md)"
    ;;
esac

info "PDF -> PS"
pdftops "$PDF_PATH" "$PS_PATH"
[[ -f "$PS_PATH" ]] || die "Failed to create PostScript file."

info "Printing on $PRINTER"
lp -d "$PRINTER" "$PS_PATH"

info "Done."
EOF
chmod +x /home/x232886/PycharmProjects/PV080_biometrics/printer/print/print_pv080.sh