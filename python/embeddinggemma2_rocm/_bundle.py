"""Content identity for runtime files in a prepared bundle."""

import hashlib
import json
from pathlib import Path


def bundle_digest(directory):
    root = Path(directory).resolve()
    files = {root / "index.json"}
    index = json.loads((root / "index.json").read_text())
    for profile in index.get("profiles", []):
        base = root / profile["path"]
        files.add(base / "bundle.json")
        config = json.loads((base / "bundle.json").read_text())
        files.add(base / config["backbone"]["path"])
        files.update(base / tensor["file"] for tensor in config["tensors"].values())
        vision = config["vision"]
        if vision["kind"] == "compiled":
            files.add(base / vision["path"])
        else:
            segmented = base / vision["path"]
            files.add(segmented / "manifest.json")
            manifest = json.loads((segmented / "manifest.json").read_text())
            files.update(segmented / stage["path"] for stage in manifest["stages"])
            files.update(
                segmented / tensor["file"]
                for tensor in manifest["weight_files"].values()
            )
    files.update(root / profile["path"] for profile in index.get("text_profiles", []))
    if index.get("tokenizer"):
        files.add(root / index["tokenizer"])
    if index.get("schema") == 2:
        files.update(root / spec["file"] for spec in index["shared"]["weights"].values())
        files.update(Path(str(path)+".memory.json") for path in list(files) if path.suffix == ".mxr")
    digest = hashlib.sha256()
    for path in sorted(files):
        relative = path.resolve().relative_to(root).as_posix()
        with path.open("rb") as stream:
            content = hashlib.file_digest(stream, "sha256").digest()
        digest.update(relative.encode() + b"\0" + content)
    return digest.hexdigest()
