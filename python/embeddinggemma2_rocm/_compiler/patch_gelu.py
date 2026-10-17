"""Replace only matching compiled BF16 gate/up GELU code objects; retain all other nodes."""

import argparse, hashlib, json
from pathlib import Path
import msgpack

p = argparse.ArgumentParser()
p.add_argument("--batch", type=int, default=16)
p.add_argument("--round", action="store_true")
p.add_argument("--source", required=True)
p.add_argument("--binary", required=True)
p.add_argument("--output", required=True)
p.add_argument("--proof")
a = p.parse_args()
source = Path(a.source)
binary = Path(a.binary)
v = msgpack.unpackb(source.read_bytes(), raw=False, strict_map_key=False)
assert v["version"] == 7, (
    "Review serialization compatibility before patching a different format"
)
code = list(binary.read_bytes())
count = 0
for node in v["modules"]["main"]["nodes"]:
    if node["name"] != "gpu::code_object":
        continue
    op = node["operator"]
    if op["symbol_name"] != "mul_mul_mul_add_mul_tanh_add_mul_mul_mul_kernel":
        continue
    assert (
        op["output"]["lens"] == [a.batch, 594, 3072]
        and op["output"]["type"] == "bf16_type"
    )
    assert len(op["expected_inputs"]) == 3 and op["output_arg"] == -1
    assert [s["strides"] for s in op["expected_inputs"]] == [
        [594 * 6144, 6144, 1],
        [594 * 6144, 6144, 1],
        [594 * 3072, 3072, 1],
    ]
    assert all(
        s["lens"] == [a.batch, 594, 3072] and s["type"] == "bf16_type"
        for s in op["expected_inputs"]
    )
    op["code_object"] = code
    op["symbol_name"] = "hip_gelu_vec8_round" if a.round else "hip_gelu_vec8"
    op["local"] = 256
    op["global"] = ((a.batch * 594 * 3072 // 8 + 255) // 256) * 256
    count += 1
assert count == 16, count
suffix = "gelu-round" if a.round else "gelu"
target = Path(a.output)
target.write_bytes(msgpack.packb(v, use_bin_type=True))
record = {
    "source": str(source),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "hsaco_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    "modified_nodes": count,
    "round_gelu_intermediate": a.round,
    "target": str(target),
}
target.with_suffix(".json").write_text(json.dumps(record, indent=2))
print(record)
