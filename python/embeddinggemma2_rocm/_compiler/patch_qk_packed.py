"""Replace selected 16-lane norm/RoPE code objects with packed-I/O kernels."""

import argparse, hashlib, json, msgpack
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--batch", type=int, default=16)
p.add_argument("--block", type=int, choices=(64, 128, 256, 512), default=256)
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
        or n["operator"]["symbol_name"] != "hip_norm_rope_l16"
    ):
        continue
    op = n["operator"]
    assert op["output_arg"] == -1 and len(op["expected_inputs"]) == 5
    assert op["output"]["lens"] == [a.batch, a.patches, 12, 64] and op["output"][
        "strides"
    ] == [a.patches * 768, 768, 64, 1]
    assert op["expected_inputs"][0]["strides"] == [a.patches * 2304, 2304, 64, 1]
    assert op["expected_inputs"][1]["type"] == "float_type" and op["expected_inputs"][
        1
    ]["strides"] == [0, 0, 0, 1]
    assert all(s["strides"] == [a.patches * 64, 64, 0, 1] for s in op["expected_inputs"][2:4])
    assert op["global"] >= a.batch * a.patches * 12 * 16
    op.update(code_object=code, symbol_name="hip_norm_rope_packed4", local=a.block,
        **{"global": ((a.batch*a.patches*12*16+a.block-1)//a.block)*a.block})
    count += 1
assert count == 32
target = Path(a.output)
target.write_bytes(msgpack.packb(v, use_bin_type=True))
r = dict(
    batch=a.batch,
    source=str(source),
    target=str(target),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    hsaco_sha256=hashlib.sha256(bytes(code)).hexdigest(),
    modified_nodes=count,
    block=a.block,
)
target.with_suffix(".json").write_text(json.dumps(r, indent=2))
print(r)
