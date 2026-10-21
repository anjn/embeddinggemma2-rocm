"""Substitute the selected GELU vector layout in an already fused Vision graph."""

import argparse, hashlib, json, msgpack
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--batch", type=int, default=16)
p.add_argument("--vector", type=int, choices=(1, 2, 4, 8, 16), default=4)
p.add_argument("--block", type=int, choices=(64,128,256,512), default=256)
p.add_argument("--source", required=True)
p.add_argument("--binary", required=True)
p.add_argument("--output", required=True)
p.add_argument("--proof")
p.add_argument("--patches", type=int, default=594)
a = p.parse_args()
source = Path(a.source)
binary = Path(a.binary)
code = list(binary.read_bytes())
v = msgpack.unpackb(source.read_bytes(), raw=False)
assert v["version"] == 7
count = 0
for n in v["modules"]["main"]["nodes"]:
    if (
        n["name"] != "gpu::code_object"
        or n["operator"]["symbol_name"] != "hip_gelu_vec8"
    ):
        continue
    op = n["operator"]
    assert op["output_arg"] == -1 and op["output"]["lens"] == [a.batch, a.patches, 3072]
    assert [s["strides"] for s in op["expected_inputs"]] == [
        [a.patches * 6144, 6144, 1],
        [a.patches * 6144, 6144, 1],
        [a.patches * 3072, 3072, 1],
    ]
    assert all(s["type"] == "bf16_type" for s in op["expected_inputs"])
    op.update(
        code_object=code,
        symbol_name=f"hip_gelu_v{a.vector}_r0",
        local=a.block,
        **{
            "global": (((a.batch * a.patches) * 3072 // a.vector + a.block - 1) // a.block)
            * a.block
        },
    )
    count += 1
assert count == 16
target = Path(a.output)
target.write_bytes(msgpack.packb(v, use_bin_type=True))
r = dict(
    batch=a.batch,
    source=str(source),
    target=str(target),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    hsaco_sha256=hashlib.sha256(bytes(code)).hexdigest(),
    modified_nodes=count,
    vector=a.vector,
    block=a.block,
)
target.with_suffix(".json").write_text(json.dumps(r, indent=2))
print(r)
