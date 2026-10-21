"""Verify compiled tuple RoPE output order using independently generated operands.
Only metadata is relabeled half for Python bindings; GPU arithmetic remains BF16.
"""

import argparse, copy, hashlib, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch, msgpack, migraphx
import ctypes as C

hip = C.CDLL("libamdhip64.so.7")
hip.hipMemcpyAsync.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int, C.c_void_p]
hip.hipMemcpyAsync.restype = C.c_int
p = argparse.ArgumentParser()
p.add_argument("--batch", type=int, default=32)
p.add_argument("--source", required=True)
p.add_argument("--output", required=True)
p.add_argument("--patches", type=int, default=594)
a = p.parse_args()
torch.set_num_threads(4)
torch.manual_seed(541)
source = Path(a.source)
v = msgpack.unpackb(source.read_bytes(), raw=False)
nodes = v["modules"]["main"]["nodes"]
m = {n["output"]: n for n in nodes}
op = copy.deepcopy(
    next(
        n
        for n in nodes
        if n["name"] == "gpu::code_object"
        and n["operator"]["symbol_name"] == "mul_mul_add_mul_mul_add_kernel"
    )
)
assert len(op["inputs"]) == 7
code_hash = hashlib.sha256(bytes(op["operator"]["code_object"])).hexdigest()
alloc = copy.deepcopy(m[op["inputs"][-1]])
scratch = copy.deepcopy(m[alloc["inputs"][0]])
unit = [copy.deepcopy(nodes[0]), scratch]
tensors = []
for i, ref in enumerate(op["inputs"][:-1]):
    shape = copy.deepcopy(m[ref]["shape"])
    size = 1 + sum((d - 1) * st for d, st in zip(shape["lens"], shape["strides"]))
    t = (
        torch.randn(size, device="cuda")
        .bfloat16()
        .as_strided(shape["lens"], shape["strides"])
    )
    tensors.append(t)
    unit.append(
        dict(
            output=f"arg{i}",
            name="@param",
            shape=shape,
            normalized=False,
            operator={"parameter": f"arg{i}"},
            inputs=[],
        )
    )
op["inputs"] = [f"arg{i}" for i in range(6)] + [alloc["output"]]
unit.extend([alloc, op])
for i in range(2):
    unit.append(
        dict(
            output=f"out{i}",
            name="get_tuple_elem",
            shape=copy.deepcopy(op["shape"]["sub_shapes"][i]),
            normalized=False,
            operator={"index": i},
            inputs=[op["output"]],
        )
    )
unit.append(
    dict(
        output="return",
        name="@return",
        shape=copy.deepcopy(op["shape"]),
        normalized=False,
        operator=[],
        inputs=["out0", "out1"],
    )
)
v["modules"]["main"]["nodes"] = unit


def relabel(x):
    if isinstance(x, dict):
        if x.get("type") == "bf16_type":
            x["type"] = "half_type"
        for val in x.values():
            relabel(val)
    elif isinstance(x, list):
        for val in x:
            relabel(val)


relabel(v)
path = source.with_name("tuple-rope-test.mxr")
path.write_bytes(msgpack.packb(v, use_bin_type=True))
program = migraphx.load(str(path))
shapes = program.get_parameter_shapes()
args = {
    f"arg{i}": migraphx.argument_from_pointer(shapes[f"arg{i}"], t.data_ptr())
    for i, t in enumerate(tensors)
}
outputs = program.run_async(
    args, torch.cuda.current_stream().cuda_stream, "ihipStream_t"
)
assert len(outputs) == 2
errors = []
for index, base in ((0, 5), (1, 2)):
    out = torch.empty(a.batch, a.patches, 12, 64, device="cuda", dtype=torch.bfloat16)
    assert (
        hip.hipMemcpyAsync(
            out.data_ptr(),
            outputs[index].data_ptr(),
            out.numel() * 2,
            3,
            torch.cuda.current_stream().cuda_stream,
        )
        == 0
    )
    rotated = tensors[4 if base == 5 else 0]
    normal = tensors[base]
    sin = tensors[1]
    cos = tensors[3]
    refs = [
        (normal.float() * cos.float() + rotated.float() * sin.float()).bfloat16(),
        normal * cos + rotated * sin,
    ]
    err = [float((out - r).abs().max()) for r in refs]
    torch.testing.assert_close(out, refs[err.index(min(err))], atol=0.0625, rtol=0.016)
    errors.append(err)
report = dict(
    batch=a.batch,
    code_sha256=code_hash,
    tuple_index_to_normal_input={"0": 5, "1": 2},
    max_abs_to_final_and_intermediate_round=errors,
    passed=True,
)
Path(a.output).write_text(json.dumps(report, indent=2))
print(report)
