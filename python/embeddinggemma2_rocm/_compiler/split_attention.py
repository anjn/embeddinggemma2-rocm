"""Split compiled Vision around attention+projection, sharing its original arena.
No GPU arithmetic is recompiled. Half metadata only exposes raw BF16 buffers to Python.
"""

import argparse, copy, hashlib, json, msgpack, numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--ffn-up", action="store_true")
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--source", required=True)
parser.add_argument("--output-dir", required=True)
cli = parser.parse_args()
source = Path(cli.source)
v = msgpack.unpackb(source.read_bytes(), raw=False)
nodes = v["modules"]["main"]["nodes"]
mp = {n["output"]: n for n in nodes}
position = {n["output"]: i for i, n in enumerate(nodes)}
assert v["version"] == 7
assert next(
    n["shape"]["lens"]
    for n in nodes
    if n["name"] == "@param" and n["operator"]["parameter"] == "pixel_values"
) == [cli.batch, 594, 768]
allowed = {
    "@param",
    "@return",
    "broadcast",
    "gpu::code_object",
    "hip::hip_allocate_memory",
    "hip::hip_copy_literal",
    "load",
    "multibroadcast",
    "reshape_lazy",
    "slice",
    "squeeze",
}
assert all(
    n["name"] in allowed or n["name"].startswith("check_context") for n in nodes
), "review non-opaque arithmetic before relabeling metadata"
original_shapes = {n["output"]: copy.deepcopy(n["shape"]) for n in nodes}
root = (
    Path(cli.output_dir)
    if cli.output_dir
    else source.parent / ("sdpa-ffn-segments" if cli.ffn_up else "sdpa-segments")
)
root.mkdir(exist_ok=True)
pairs = []
for n in nodes:
    if n["name"] == "gpu::code_object" and n["operator"]["symbol_name"].startswith(
        "mlir_transpose_transpose_transpose_dot"
    ):
        proj = next(
            x
            for x in nodes
            if x["name"] == "gpu::code_object"
            and x["operator"]["symbol_name"] == "mlir_transpose_reshape_dot"
            and x["inputs"][0] == n["output"]
        )
        pairs.append((n, proj))
assert len(pairs) == 16
operations = [("attention", a, p) for a, p in pairs]
if cli.ffn_up:
    up = [
        n
        for n in nodes
        if n["name"] == "gpu::code_object"
        and n["operator"]["symbol_name"] == "mlir_dot"
        and n["operator"]["expected_inputs"][1]["lens"] == [768, 6144]
    ]
    assert len(up) == 16
    operations.extend(("linear", n, n) for n in up)
operations.sort(key=lambda op: position[op[2]["output"]])
allocs = {
    n["output"]: n["shape"] for n in nodes if n["name"] == "hip::hip_allocate_memory"
}
params = {
    n["output"]: dict(shape=n["shape"], parameter=n["operator"]["parameter"])
    for n in nodes
    if n["name"] == "@param"
}


def location(ref):
    n = mp[ref]
    name = n["name"]
    if name in ("hip::hip_allocate_memory", "@param"):
        return ref, 0
    if name == "gpu::code_object":
        return location(n["inputs"][n["operator"]["output_arg"]])
    if name == "load":
        base, offset = location(n["inputs"][0])
        return base, offset + n["operator"]["offset"]
    if name == "slice":
        parent = mp[n["inputs"][0]]
        base, offset = location(parent["output"])
        size = {"bf16_type": 2, "half_type": 2, "float_type": 4, "int8_type": 1}[
            parent["shape"]["type"]
        ]
        return base, offset + size * sum(
            start * parent["shape"]["strides"][axis]
            for start, axis in zip(n["operator"]["starts"], n["operator"]["axes"])
        )
    if name in (
        "reshape_lazy",
        "multibroadcast",
        "broadcast",
        "transpose",
        "squeeze",
        "unsqueeze",
        "identity",
    ):
        return location(n["inputs"][0])
    raise ValueError((ref, name))


def describe(ref):
    base, offset = location(ref)
    return dict(base=base, offset=offset, shape=copy.deepcopy(mp[ref]["shape"]))


def parameter(ref):
    return dict(
        output=ref,
        name="@param",
        shape=copy.deepcopy(mp[ref]["shape"]),
        normalized=False,
        operator={"parameter": ref},
        inputs=[],
    )


def relabel_shape(s):
    if s.get("type") == "bf16_type":
        s["type"] = "half_type"
    for sub in s.get("sub_shapes", []):
        relabel_shape(sub)


def relabel_node(n):
    relabel_shape(n["shape"])
    op = n["operator"]
    if isinstance(op, dict):
        for key in ("shape", "output"):
            if key in op:
                relabel_shape(op[key])
        for s in op.get("expected_inputs", []):
            relabel_shape(s)
        if "literal" in op:
            relabel_shape(op["literal"]["shape"])


weights = {}
events = []
removed = {n["output"] for op in operations for n in op[1:]}
stages = []
start = 0
ends = [position[op[2]["output"]] for op in operations] + [len(nodes) - 1]
for stage, end in enumerate(ends):
    selected = {}
    external = {}

    def visit(ref):
        if ref in selected:
            return
        n = mp[ref]
        if (
            n["name"] == "hip::hip_allocate_memory"
            or n["name"] == "@param"
            or (n["name"] == "gpu::code_object" and position[ref] < start)
        ):
            selected[ref] = parameter(ref)
            external[ref] = describe(ref)
            return
        assert ref not in removed, (
            "unexpected dependency on replaced attention",
            stage,
            ref,
        )
        for dep in n["inputs"]:
            visit(dep)
        selected[ref] = copy.deepcopy(n)

    roots = [
        n["output"]
        for n in nodes[start : end + 1]
        if n["name"] == "gpu::code_object" and n["output"] not in removed
    ]
    for ref in roots:
        visit(ref)
    result = nodes[-1]["inputs"][0] if stage == len(operations) else roots[-1]
    visit(result)
    ordered = [selected[n["output"]] for n in nodes if n["output"] in selected]
    ordered = [copy.deepcopy(nodes[0])] + [
        n for n in ordered if n["output"] != nodes[0]["output"]
    ]
    ordered.append(
        dict(
            output=f"segment_return_{stage}",
            name="@return",
            shape=copy.deepcopy(mp[result]["shape"]),
            normalized=False,
            operator=[],
            inputs=[result],
        )
    )
    known = set()
    for n in ordered:
        assert set(n["inputs"]) <= known, (stage, n["output"], set(n["inputs"]) - known)
        known.add(n["output"])
        relabel_node(n)
    unit = {k: copy.deepcopy(value) for k, value in v.items() if k != "modules"}
    unit["modules"] = {"main": {"name": "main", "nodes": ordered}}
    path = root / f"segment-{stage:02}.mxr"
    path.write_bytes(msgpack.packb(unit, use_bin_type=True))
    stages.append(dict(path=path.name, external=external, nodes=len(ordered)))
    if stage < len(operations):
        kind, first, last = operations[stage]
        wn = mp[last["inputs"][1]]
        literal = wn["operator"]["literal"]
        shape = literal["shape"]
        assert shape["type"] == "bf16_type" and shape["strides"] == [
            shape["lens"][1],
            1,
        ]
        raw = literal["data"]
        assert len(raw) == 1 and isinstance(raw[0], bytes)
        weights[f"w{stage}"] = (
            np.frombuffer(raw[0], dtype=np.uint16).reshape(shape["lens"]).copy()
        )
        tensors = (
            dict(
                q=describe(first["inputs"][0]),
                k=describe(first["inputs"][1]),
                v=describe(first["inputs"][2]),
                output=describe(last["output"]),
            )
            if kind == "attention"
            else dict(x=describe(first["inputs"][0]), output=describe(last["output"]))
        )
        events.append(dict(kind=kind, tensors=tensors, weight=f"w{stage}"))
    start = end + 1
assert {n["output"]: n["shape"] for n in nodes} == original_shapes, (
    "metadata relabel mutated source shapes"
)
assert all(
    d["shape"]["type"] == "bf16_type"
    for event in events
    for d in event["tensors"].values()
)
np.savez(root / "weights.npz", **weights)
report = dict(
    batch=cli.batch,
    source=str(source),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    allocs=allocs,
    params=params,
    stages=stages,
    events=events,
    ffn_up=cli.ffn_up,
    output=describe(nodes[-1]["inputs"][0]),
    metadata_relabel="bf16 to half only, raw GPU storage/code unchanged",
)
(root / "manifest.json").write_text(json.dumps(report, indent=2))
print("wrote", len(stages), "segments", sum(x["nodes"] for x in stages), "nodes")
