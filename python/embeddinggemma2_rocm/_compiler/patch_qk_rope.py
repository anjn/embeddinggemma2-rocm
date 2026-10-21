"""Fuse Q/K norm + axial RoPE in 15 layers, optionally including layer zero.
New dedicated shared work buffers avoid extending lifetimes in the original aliased arena.
"""

import argparse, hashlib, json, msgpack
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--batch", type=int, default=16)
p.add_argument("--round", action="store_true")
p.add_argument("--gelu-round", action="store_true")
p.add_argument("--all-layers", action="store_true")
p.add_argument("--lanes", type=int, choices=(4, 8, 16, 32, 64), default=32)
p.add_argument("--source", required=True)
p.add_argument("--binary", required=True)
p.add_argument("--output", required=True)
p.add_argument("--proof")
p.add_argument("--patches", type=int, default=594)
p.add_argument("--block", type=int, choices=(64,128,256,512), default=128)
a = p.parse_args()
assert a.lanes == 32 or a.round
source = Path(a.source)
v = msgpack.unpackb(source.read_bytes(), raw=False)
assert v["version"] == 7
nodes = v["modules"]["main"]["nodes"]
mp = {n["output"]: n for n in nodes}
code = list(Path(a.binary).read_bytes())
shapeq = {
    "type": "bf16_type",
    "sub_shapes": [],
    "lens": [a.batch, a.patches, 12, 64],
    "strides": [a.patches * 768, 768, 64, 1],
    "dynamic_dimensions": [],
}
shapez = {**shapeq, "lens": [a.batch, a.patches, 2304], "strides": [a.patches * 2304, 2304, 1]}
alloc = []
for name, shape in [
    ("hip_fused_z", shapez),
    ("hip_fused_q", shapeq),
    ("hip_fused_k", shapeq),
]:
    n = dict(
        output=name,
        name="hip::hip_allocate_memory",
        shape=shape,
        normalized=False,
        operator={"shape": shape, "id": "main:" + name},
        inputs=[],
    )
    alloc.append(n)
    mp[name] = n
norm_name = "convert_mul_mul_reduce_sum_add_sqrt_div_mul_mul_convert_kernel"


def find_norm(ref, visited=None):
    if visited is None:
        visited = set()
    if ref in visited:
        return set()
    visited.add(ref)
    n = mp[ref]
    if n["name"] == "gpu::code_object":
        if n["operator"]["symbol_name"] == norm_name and n["shape"] == shapeq:
            return {ref}
        if "dot" in n["operator"]["symbol_name"]:
            return set()
    out = set()
    for i in n["inputs"]:
        out.update(find_norm(i, visited))
    return out


count = 0
gemms = set()
for attn in nodes:
    if attn["name"] != "gpu::code_object" or not attn["operator"][
        "symbol_name"
    ].startswith("mlir_transpose_transpose_transpose_dot"):
        continue
    for index, ref in enumerate(attn["inputs"][:2]):
        n = mp[ref]
        if n["name"] == "get_tuple_elem":
            if not a.all_layers:
                continue
            parent = mp[n["inputs"][0]]
            assert (
                parent["operator"]["symbol_name"] == "mul_mul_add_mul_mul_add_kernel"
                and len(parent["inputs"]) == 7
            )
            # This compiled tuple kernel returns K at index 0 and Q at index 1.
            # Refuse a recompiled tuple kernel until its output order is revalidated.
            tuple_hash = hashlib.sha256(
                bytes(parent["operator"]["code_object"])
            ).hexdigest()
            if (
                tuple_hash
                != "89b2fa4728b50a65194cfa544ecd9fa18e56a82adb1ea28aa240684a1c6e19fb"
            ):
                proof = json.loads(Path(a.proof).read_text())
                assert (
                    proof["passed"]
                    and proof["batch"] == a.batch
                    and proof["code_sha256"] == tuple_hash
                )
                assert proof["tuple_index_to_normal_input"] == {"0": 5, "1": 2}
            elem = n["operator"]["index"]
            assert elem in (0, 1)
            norms = find_norm(parent["inputs"][2 if elem == 1 else 5])
            sin, cos = parent["inputs"][1], parent["inputs"][3]
            n["name"] = "gpu::code_object"
        else:
            assert (
                n["name"] == "gpu::code_object"
                and n["shape"] == shapeq
                and len(n["inputs"]) == 8
            )
            norms = find_norm(ref)
            sin, cos = n["inputs"][4], n["inputs"][6]
        assert len(norms) == 1, norms
        norm = mp[next(iter(norms))]
        raw, w = norm["inputs"][:2]
        current = raw
        while mp[current]["name"] in ("reshape_lazy", "slice"):
            current = mp[current]["inputs"][0]
        gemm = mp[current]
        assert gemm["name"] == "gpu::code_object" and gemm["shape"] == shapez
        assert gemm["operator"]["output_arg"] == 2 and len(gemm["inputs"]) == 3
        gemm["inputs"][-1] = "hip_fused_z"
        gemms.add(current)
        for coeff in (sin, cos):
            assert mp[coeff]["shape"]["strides"] == [a.patches * 64, 64, 0, 1] and mp[coeff][
                "shape"
            ]["lens"] == [a.batch, a.patches, 12, 64]
        args = [raw, w, cos, sin, "hip_fused_q" if index == 0 else "hip_fused_k"]
        n["inputs"] = args
        n["operator"] = {
            "code_object": code,
            "symbol_name": f"hip_norm_rope_l{a.lanes}",
            "global": ((a.batch * a.patches * 12 * a.lanes + a.block - 1) // a.block) * a.block,
            "local": a.block,
            "expected_inputs": [mp[i]["shape"] for i in args],
            "output": shapeq,
            "output_arg": -1,
        }
        count += 1
assert count == (32 if a.all_layers else 30) and len(gemms) == (
    16 if a.all_layers else 15
), (count, len(gemms))
# Mark live data dependencies; preserve the compiled context guard and public parameters.
live = set()


def mark(ref):
    if ref in live:
        return
    live.add(ref)
    for i in mp[ref]["inputs"]:
        mark(i)


for n in nodes:
    if n["name"] in ("@return", "@param") or n["name"].startswith("check_context"):
        mark(n["output"])
newnodes = [n for n in nodes if n["output"] in live]
newnodes[1:1] = [n for n in alloc if n["output"] in live]
known = set()
for n in newnodes:
    assert set(n["inputs"]) <= known, (n["output"], set(n["inputs"]) - known)
    known.add(n["output"])
v["modules"]["main"]["nodes"] = newnodes
suffix = "qk-rope-round" if a.round else "qk-rope"
suffix += "-all" if a.all_layers else ""
suffix += f"-l{a.lanes}" if a.lanes != 32 else ""
out = Path(a.output)
out.write_bytes(msgpack.packb(v, use_bin_type=True))
record = {
    "batch": a.batch,
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "hsaco_sha256": hashlib.sha256(bytes(code)).hexdigest(),
    "target": str(out),
    "fused_qk_nodes": count,
    "layers": len(gemms),
    "old_nodes": len(nodes),
    "new_nodes": len(newnodes),
    "extra_workspace_bytes": (a.batch * a.patches) * (2304 + 768 * 2) * 2,
    "round_products": a.round,
    "lanes": a.lanes,
}
out.with_suffix(".json").write_text(json.dumps(record, indent=2))
print(record)
