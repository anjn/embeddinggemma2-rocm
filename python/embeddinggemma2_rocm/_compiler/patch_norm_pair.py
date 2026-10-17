"""Fuse consecutive residual RMSNorm and unweighted RMSNorm, preserving both outputs."""

import argparse, copy, hashlib, json, msgpack
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--source", required=True)
p.add_argument("--binary", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
source = Path(a.source)
binary = Path(a.binary)
v = msgpack.unpackb(source.read_bytes(), raw=False)
nodes = v["modules"]["main"]["nodes"]
mp = {n["output"]: n for n in nodes}
first_symbol = "convert_mul_mul_reduce_sum_add_sqrt_div_mul_mul_convert_add_kernel"
b = "convert_mul_mul_reduce_sum_add_sqrt_div_mul_convert_kernel"
pairs = []
for second in nodes:
    if (
        second["name"] != "gpu::code_object"
        or second["operator"]["symbol_name"] != b
        or second["shape"]["lens"] != [16, 594, 768]
    ):
        continue
    first = mp[second["inputs"][0]]
    if (
        first["name"] == "gpu::code_object"
        and first["operator"]["symbol_name"] == first_symbol
    ):
        pairs.append((first, second))
assert len(pairs) == 31, len(pairs)
insert = {}
alias = {}
code = list(binary.read_bytes())
for first, second in pairs:
    out1 = mp[first["inputs"][-1]]
    out2 = mp[second["inputs"][-1]]
    assert out1["name"] == out2["name"] == "load" and out1["inputs"] == out2["inputs"]
    size = 16 * 594 * 768 * 2
    assert abs(out1["operator"]["offset"] - out2["operator"]["offset"]) >= size
    pre = copy.deepcopy(out2)
    pre["output"] = "norm_pair_dest_" + second["output"]
    insert[first["output"]] = pre
    first["inputs"].append(pre["output"])
    op = first["operator"]
    op["expected_inputs"].append(copy.deepcopy(second["shape"]))
    op.update(
        code_object=code,
        symbol_name="hip_norm_pair",
        output_arg=3,
        local=256,
        **{"global": ((16 * 594 * 32 + 255) // 256) * 256},
    )
    al = copy.deepcopy(out2)
    al["output"] = second["output"]
    alias[second["output"]] = al
new = []
for n in nodes:
    if n["output"] in insert:
        new.append(insert[n["output"]])
    new.append(alias.get(n["output"], n))
known = set()
for n in new:
    assert set(n["inputs"]) <= known, (n["output"], set(n["inputs"]) - known)
    known.add(n["output"])
v["modules"]["main"]["nodes"] = new
target = Path(a.output)
target.write_bytes(msgpack.packb(v, use_bin_type=True))
report = dict(
    source=str(source),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    hsaco_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
    target=str(target),
    fused_pairs=len(pairs),
    bf16_boundaries="post-norm before residual add; residual result before next norm; final norm result",
)
target.with_suffix(".json").write_text(json.dumps(report, indent=2))
print(report)
