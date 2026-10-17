"""Download the fixed uncropped Flickr8k retrieval sample into ignored artifacts."""

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps

REPOSITORY = "intro/flickr8k"
REVISION = "c7a02cb487da0eca729e133194b0f842cb578c08"
SEED = 20261016


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/quality-data")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.output)
    (root / "resized").mkdir(parents=True, exist_ok=True)

    def download(filename):
        return Path(
            hf_hub_download(
                REPOSITORY,
                filename,
                repo_type="dataset",
                revision=REVISION,
                local_files_only=args.local_files_only,
            )
        )

    with download("test/metadata.csv").open() as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: row["file_name"])
    if len(rows) != 1000:
        raise ValueError("Unexpected pinned test split size")
    indices = np.sort(np.random.default_rng(SEED).choice(len(rows), 256, replace=False))

    def prepare(index):
        row = rows[index]
        source = download("test/" + row["file_name"])
        name = Path(row["file_name"]).stem
        relative = Path("resized") / (name + ".png")
        with Image.open(source) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            size = list(image.size)
            content = ImageOps.contain(image, (528, 288), Image.Resampling.BICUBIC)
            frame = Image.new("RGB", (528, 288), (127, 127, 127))
            frame.paste(
                content, ((528 - content.width) // 2, (288 - content.height) // 2)
            )
            frame.save(root / relative)
        return dict(
            id=name,
            test_row_index=int(index),
            image=relative.as_posix(),
            original_size=size,
            original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            captions=[row[f"caption_{i}"] for i in range(5)],
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        records = []
        for count, record in enumerate(pool.map(prepare, indices), 1):
            records.append(record)
            if count % 32 == 0:
                print(f"Prepared {count}/256", flush=True)
    if len({row["id"] for row in records}) != 256 or any(
        not caption.strip() for row in records for caption in row["captions"]
    ):
        raise ValueError("Invalid image IDs or captions")
    manifest = dict(
        dataset=REPOSITORY,
        revision=REVISION,
        split="test",
        sample_size=256,
        test_split_size=1000,
        seed=SEED,
        preprocessing="EXIF transpose; RGB; fit inside 528x288 with bicubic; gray letterbox without cropping. Evaluator prescales to 512x288 with bilinear.",
        images=records,
    )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(root / "manifest.json")


if __name__ == "__main__":
    main()
