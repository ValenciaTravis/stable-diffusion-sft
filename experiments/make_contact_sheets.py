#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def wrap_label(text: str, width: int = 34) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)[:4])


def task_sort_key(record: dict[str, Any]) -> tuple:
    return (
        record.get("experiment_id", ""),
        record.get("variant_id", ""),
        record.get("prompt_id", ""),
        record.get("seed", 0),
        record.get("lora_scale", 0.0),
    )


def make_sheet(records: list[dict[str, Any]], output_path: Path, columns: int, thumb_size: int, label_height: int) -> None:
    font = load_font(14)
    title_font = load_font(15)
    rows = math.ceil(len(records) / columns)
    cell_w = thumb_size
    cell_h = thumb_size + label_height
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)

    for idx, record in enumerate(records):
        image_path = Path(record["output_path"])
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.contain(image, (thumb_size, thumb_size), method=Image.Resampling.LANCZOS)

        x = (idx % columns) * cell_w
        y = (idx // columns) * cell_h
        image_x = x + (thumb_size - image.width) // 2
        image_y = y
        sheet.paste(image, (image_x, image_y))

        label = (
            f"{record.get('variant_id', '')}\n"
            f"{record.get('prompt_id', '')} seed={record.get('seed')} scale={record.get('lora_scale')}"
        )
        if record.get("metadata", {}).get("checkpoint"):
            label += f"\n{record['metadata']['checkpoint']}"
        if record.get("metadata", {}).get("merge_label"):
            label += f"\n{record['metadata']['merge_label']}"
        draw.text((x + 6, y + thumb_size + 4), wrap_label(label), fill="black", font=font)

    title = output_path.stem.replace("_", " ")
    draw.text((8, 6), title, fill="black", font=title_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def write_index(grouped: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    lines = ["# Contact Sheets", ""]
    for experiment_id in sorted(grouped):
        image_name = f"{experiment_id}.jpg"
        lines.append(f"- `{experiment_id}`: [{image_name}]({image_name}) ({len(grouped[experiment_id])} images)")
    (output_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build contact sheets from report experiment metadata.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--thumb-size", type=int, default=256)
    parser.add_argument("--label-height", type=int, default=94)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = [
        record
        for record in metadata.get("images", [])
        if record.get("status") in {"generated", "skipped_existing"} and Path(record["output_path"]).exists()
    ]
    records.sort(key=task_sort_key)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["experiment_id"]].append(record)

    for experiment_id, group in grouped.items():
        make_sheet(group, output_dir / f"{experiment_id}.jpg", args.columns, args.thumb_size, args.label_height)
        print(f"saved {output_dir / f'{experiment_id}.jpg'}")

    write_index(grouped, output_dir)
    print(f"saved {output_dir / 'index.md'}")


if __name__ == "__main__":
    main()

