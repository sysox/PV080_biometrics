#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps, ImageDraw, ImageFont

# ----------------------------
# Configuration
# ----------------------------

UCO_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")
SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PROCESSED_DIRNAME = "processed"

# A4 at 300 DPI
PAGE_W = 2480
PAGE_H = 3508

LEFT = 80
RIGHT = 80
TOP = 80
BOTTOM = 80

HEADER_H = 50
ROW_H = 310
ROW_GAP = 25

UCO_COL_W = 260
IMG_GAP = 20
IMG_BOX_H = 240
LABEL_H = 35

BG = "white"
FG = "black"

# ----------------------------
# Helpers
# ----------------------------

def die(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)

def info(msg: str) -> None:
    print(f"[INFO] {msg}")

def warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def extract_uco(path: Path) -> str | None:
    m = UCO_RE.search(path.name)
    return m.group(1) if m else None

def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(p) if p.isdigit() else p for p in parts]

def load_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()

FONT_TITLE = load_font(26)
FONT_UCO = load_font(34)
FONT_LABEL = load_font(20)
FONT_SMALL = load_font(18)

def fit_image(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    img.thumbnail((max_w, max_h))
    canvas = Image.new("L", (max_w, max_h), 255)
    x = (max_w - img.width) // 2
    y = (max_h - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas.convert("RGB")

def next_batch_number(root: Path) -> int:
    nums = set()

    # existing PDFs like 1.pdf, 2.pdf, ...
    for p in root.glob("*.pdf"):
        if p.stem.isdigit():
            nums.add(int(p.stem))

    # existing processed folders like processed/1, processed/2, ...
    processed_root = root / PROCESSED_DIRNAME
    if processed_root.exists():
        for child in processed_root.iterdir():
            if child.is_dir() and child.name.isdigit():
                nums.add(int(child.name))

    return max(nums, default=0) + 1

def prepare_batch_paths(root: Path) -> tuple[int, Path, Path]:
    n = next_batch_number(root)
    pdf_path = root / f"{n}.pdf"
    batch_dir = root / PROCESSED_DIRNAME / str(n)
    batch_dir.mkdir(parents=True, exist_ok=False)
    return n, pdf_path, batch_dir

def group_files(input_dir: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)

    for p in sorted(input_dir.rglob("*"), key=natural_key):
        if not p.is_file():
            continue
        if PROCESSED_DIRNAME in p.parts:
            continue
        if p.suffix.lower() not in SUPPORTED:
            continue

        uco = extract_uco(p)
        if not uco:
            continue

        grouped[uco].append(p)

    return grouped

def choose_students(grouped: dict[str, list[Path]]) -> tuple[list[tuple[str, list[Path]]], list[str]]:
    rows = []
    skipped = []

    for uco in sorted(grouped.keys()):
        files = sorted(grouped[uco], key=natural_key)
        if len(files) < 3:
            skipped.append(uco)
            continue
        rows.append((uco, files[:3]))

    return rows, skipped

def make_pages(rows: list[tuple[str, list[Path]]], batch_name: str) -> list[Image.Image]:
    content_h = PAGE_H - TOP - BOTTOM - HEADER_H
    rows_per_page = max(1, content_h // (ROW_H + ROW_GAP))

    img_total_w = PAGE_W - LEFT - RIGHT - UCO_COL_W
    img_box_w = (img_total_w - 2 * IMG_GAP) // 3

    pages = []

    for page_idx in range(0, len(rows), rows_per_page):
        chunk = rows[page_idx:page_idx + rows_per_page]

        page = Image.new("RGB", (PAGE_W, PAGE_H), BG)
        draw = ImageDraw.Draw(page)

        title = f"Fingerprint print batch {batch_name}"
        draw.text((LEFT, TOP), title, fill=FG, font=FONT_TITLE)

        y = TOP + HEADER_H

        for uco, files in chunk:
            uco_y = y + (IMG_BOX_H // 2) - 20
            draw.text((LEFT, uco_y), f"UCO {uco}", fill=FG, font=FONT_UCO)

            x = LEFT + UCO_COL_W

            for i in range(3):
                box_x0 = x + i * (img_box_w + IMG_GAP)
                box_y0 = y
                box_x1 = box_x0 + img_box_w
                box_y1 = box_y0 + IMG_BOX_H

                draw.rectangle([box_x0, box_y0, box_x1, box_y1], outline="black", width=2)

                src = files[i]
                try:
                    img = Image.open(src)
                    fitted = fit_image(img, img_box_w - 12, IMG_BOX_H - 12)
                    page.paste(fitted, (box_x0 + 6, box_y0 + 6))
                    label = src.stem[:28]
                except Exception as e:
                    label = "ERROR"
                    draw.text((box_x0 + 8, box_y0 + 8), str(e)[:40], fill=FG, font=FONT_SMALL)

                label_y = box_y1 + 6
                draw.text((box_x0 + 3, label_y), label, fill=FG, font=FONT_LABEL)

            y += ROW_H + ROW_GAP

        pages.append(page)

    return pages

def move_used_files(rows: list[tuple[str, list[Path]]], batch_dir: Path) -> None:
    for uco, files in rows:
        student_dir = batch_dir / uco
        student_dir.mkdir(parents=True, exist_ok=True)

        for src in files:
            dst = student_dir / src.name

            if dst.exists():
                stem = src.stem
                suf = src.suffix
                k = 2
                while True:
                    alt = student_dir / f"{stem}_{k}{suf}"
                    if not alt.exists():
                        dst = alt
                        break
                    k += 1

            shutil.move(str(src), str(dst))

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input_dir>")
        sys.exit(1)

    input_dir = Path(sys.argv[1]).resolve()
    if not input_dir.is_dir():
        die(f"Input directory does not exist: {input_dir}")

    grouped = group_files(input_dir)
    if not grouped:
        die("No supported image files with 6-digit UCO found.")

    rows, skipped = choose_students(grouped)
    if not rows:
        die("No students with at least 3 fingerprint files found.")

    batch_num, output_pdf, batch_dir = prepare_batch_paths(input_dir)

    info(f"Batch number: {batch_num}")
    info(f"Output PDF: {output_pdf}")
    info(f"Processed folder: {batch_dir}")
    info(f"Students selected: {len(rows)}")

    if skipped:
        warn(f"Skipped students with fewer than 3 files: {', '.join(skipped)}")

    pages = make_pages(rows, batch_name=str(batch_num))
    first, rest = pages[0], pages[1:]
    first.save(output_pdf, save_all=True, append_images=rest, resolution=300.0)

    move_used_files(rows, batch_dir)

    info(f"Created PDF: {output_pdf}")
    info(f"Moved used files to: {batch_dir}")

if __name__ == "__main__":
    main()