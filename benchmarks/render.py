"""Aggregate complete reversed rounds; never turn missing/failed runs into speed values."""

import argparse, collections, json, statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("directory")
p.add_argument("--output", default="benchmarks/comparison.md")
a = p.parse_args()
root = Path(a.directory)
groups = collections.defaultdict(list)
for path in root.glob("*.json"):
    r = json.loads(path.read_text())
    if "runner_id" in r:
        groups[r["budget"], r["batch"], r["runner_id"]].append(r)
preferred = ["native", "torch", "llama_hip", "llama_vulkan"]
ids = sorted(
    {key[2] for key in groups},
    key=lambda x: (preferred.index(x) if x in preferred else len(preferred), x),
)
labels = {
    "native": "Native",
    "torch": "PyTorch BF16",
    "llama_hip": "llama.cpp HIP",
    "llama_vulkan": "llama.cpp Vulkan",
}
rows = []
summary = []
for budget in (70, 140, 280):
    for batch in (1, 4, 32, 128):
        cells = [str(budget), str(batch)]
        cell_hashes = set()
        token_shapes = set()
        for name in ids:
            records = groups.get((budget, batch, name), [])
            entry = dict(budget=budget, batch=batch, runner=name)
            if (
                len(records) == 2
                and {r.get("round") for r in records} == {0, 1}
                and all(r.get("status") == "ok" for r in records)
            ):
                assert all(
                    r.get("warmup", 0) >= 3
                    and r.get("repeat", 0) >= 5
                    and len(r.get("seconds", [])) >= 5
                    for r in records
                ), "Incomplete timing protocol"
                hashes = {r.get("input_sha256") for r in records}
                assert len(hashes) == 1 and None not in hashes
                assert records[0].get("metadata") == records[1].get("metadata"), (
                    f"Backend settings changed across rounds: {name}"
                )
                for r in records:
                    meta = r["metadata"]
                    execution = meta.get("capabilities", {}).get("last_execution", {})
                    tokens = meta.get(
                        "actual_image_tokens", execution.get("image_tokens")
                    )
                    sequence = meta.get(
                        "actual_sequence_length", execution.get("sequence")
                    )
                    assert tokens is not None and sequence is not None, (
                        "Missing token counts"
                    )
                    token_shapes.add((tokens, sequence))
                cell_hashes.update(hashes)
                ms = statistics.median(
                    [t * 1000 / batch for r in records for t in r["seconds"]]
                )
                entry.update(
                    status="ok",
                    ms_per_image=ms,
                    images_per_second=1000 / ms,
                    input_sha256=hashes.pop(),
                )
                cells.append(f"{ms:.2f}")
            else:
                status = (
                    "/".join(sorted({r.get("status", "invalid") for r in records}))
                    or "not measured"
                )
                if status == "ok":
                    status = "incomplete"
                entry["status"] = status
                display = (
                    "failed (139)" if "process_error" in status
                    else "excluded" if "validation_failed" in status
                    else "omitted" if "omitted" in status or status == "incomplete"
                    else status
                )
                cells.append(display)
            summary.append(entry)
        assert len(cell_hashes) <= 1, "Backends used different inputs"
        assert len(token_shapes) <= 1, "Backends used different token grids"
        rows.append("| " + " | ".join(cells) + " |")
text = (
    "| Token budget | Batch | "
    + " | ".join(labels.get(name, name) for name in ids)
    + " |\n|---:|---:|"
    + "---:|" * len(ids)
    + "\n"
    + "\n".join(rows)
    + "\n\nAll numeric cells are median ms/image across two reversed-order runs. Failed, unsupported and unmeasured cases remain explicit.\n"
)
Path(a.output).write_text(text)
Path(a.output).with_suffix(".json").write_text(json.dumps(summary, indent=2))
print(text)
