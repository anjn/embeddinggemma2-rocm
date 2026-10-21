"""Conservative, shape-guarded gfx1150/budget70 recipe derived from measured kernels."""

from pathlib import Path
import json, subprocess, sys, shutil, hashlib
import numpy as np


def mixed_vision(model):
    import torch
    from types import MethodType
    from transformers.models.gemma4.modeling_gemma4 import apply_multidimensional_rope

    def norm(self, x):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)

    def mlp(self, x):
        x = x.to(torch.bfloat16)
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    def attention(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        **kwargs,
    ):
        shape = hidden_states.shape[:-1]
        head_shape = (*shape, -1, self.head_dim)
        x = hidden_states.to(torch.bfloat16)
        q = self.q_norm(self.q_proj(x).view(head_shape))
        k = self.k_norm(self.k_proj(x).view(head_shape))
        v = self.v_norm(self.v_proj(x).view(head_shape))
        cos, sin = (item.to(torch.bfloat16) for item in position_embeddings)
        q = apply_multidimensional_rope(q, cos, sin, position_ids).transpose(1, 2)
        k = apply_multidimensional_rope(k, cos, sin, position_ids).transpose(1, 2)
        v = v.transpose(1, 2)
        score = (q @ k.transpose(-2, -1)).float() * self.scaling
        if attention_mask is not None:
            score = score + attention_mask
        prob = torch.softmax(score, dim=-1).to(torch.bfloat16)
        value = (prob @ v).transpose(1, 2).contiguous().reshape(*shape, -1)
        return self.o_proj(value.to(torch.bfloat16)), None

    def layer_forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        **kwargs,
    ):
        x = hidden_states.to(torch.bfloat16)
        value, _ = self.self_attn(
            self.input_layernorm(x),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        x = x + self.post_attention_layernorm(value)
        return x + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(x))
        )

    for module in model.vision_tower.modules():
        if module.__class__.__name__ == "Gemma4RMSNorm":
            module._norm = MethodType(norm, module)
    for layer in model.vision_tower.encoder.layers:
        layer.forward = MethodType(layer_forward, layer)
        layer.mlp.forward = MethodType(mlp, layer.mlp)
        for linear in (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj):
            linear.to(torch.bfloat16)
        layer.self_attn.forward = MethodType(attention, layer.self_attn)
        for linear in (
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.o_proj,
        ):
            linear.to(torch.bfloat16)
    model.vision_tower.patch_embedder.to(torch.bfloat16)
    model.embed_vision.embedding_projection.to(torch.bfloat16)


def optimize(source, dest, arch="gfx1150", *, batch=16, patches=594, config=None):
    """Compile and install a checked, shape-specific HIP recipe on the local GPU."""
    config = config or dict(gelu=dict(block=256,vector=4),
                            qk=dict(block=256,lanes=16,packed=True),
                            norm=dict(block=256,lanes=32), segmented=True)
    source, dest = Path(source).resolve(), Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    kernels = Path(__file__).parent / "kernels"
    hipcc = shutil.which("hipcc") or "/opt/rocm/bin/hipcc"
    binaries = {}
    for name, cpp in [("gelu","gelu_layout.cpp"),("qk","qk_rope_packed.cpp"),("norm","norm_patch.cpp")]:
        if not config.get(name): continue
        target = dest / (name+".hsaco")
        subprocess.run([hipcc,"-O3","--genco","--offload-arch="+arch,
                        "-DROWS="+str(batch*patches),str(kernels/cpp),"-o",str(target)],check=True,timeout=300)
        binaries[name] = target
    proof = dest / "tuple-proof.json"
    if config.get("qk"):
        subprocess.run([sys.executable,"-m","embeddinggemma2_rocm._compiler.prove_tuple",
                        "--source",str(source),"--batch",str(batch),"--patches",str(patches),
                        "--output",str(proof)],check=True,timeout=300)
    steps = []
    if config.get("gelu"):
        g=config["gelu"]
        steps.extend([("patch_gelu","gelu",[]),
                      ("patch_gelu_layout","gelu",["--vector",str(g["vector"]),"--block",str(g["block"])])])
    if config.get("qk"):
        q=config["qk"]
        steps.append(("patch_qk_rope","qk",["--round","--all-layers","--lanes",str(q["lanes"]),
                                             "--block",str(q["block"]),"--proof",str(proof)]))
        if q.get("packed"):
            if q["lanes"]!=16: raise ValueError("Packed QK requires 16 lanes")
            steps.append(("patch_qk_packed","qk",["--block",str(q["block"])]))
    if config.get("norm"):
        n=config["norm"]
        steps.append(("patch_norm_pair","norm",["--lanes",str(n["lanes"]),"--block",str(n["block"])]))
    current = source
    for i,(module,binary,args) in enumerate(steps):
        target=dest/f"pass-{i}.mxr"
        subprocess.run([sys.executable,"-m","embeddinggemma2_rocm._compiler."+module,
                        "--source",str(current),"--binary",str(binaries[binary]),
                        "--output",str(target),"--batch",str(batch),"--patches",str(patches),*args],check=True,timeout=300)
        current=target
    if not config.get("segmented",True): return current
    segments = dest/"segments"
    subprocess.run([sys.executable,"-m","embeddinggemma2_rocm._compiler.split_attention",
                    "--source",str(current),"--output-dir",str(segments),"--batch",str(batch),
                    "--patches",str(patches),"--ffn-up"],check=True,timeout=300)
    from ..preparation import save_tensor, write_json
    manifest=json.loads((segments/"manifest.json").read_text())
    manifest["weight_files"]={}
    with np.load(segments/"weights.npz") as weights:
        for name in weights.files:
            manifest["weight_files"][name]=save_tensor(segments,name,weights[name])
    manifest["source"]=current.name
    write_json(segments/"manifest.json",manifest)
    return segments


def static_vision(model, positions, batch=16):
    """Fixed-shape export preserving the measured gather and pooling layout."""
    import torch
    from torch import nn

    positions = positions.repeat(batch, 1, 1)

    class Pool(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = model.embed_vision
            self.scale = model.vision_tower.config.hidden_size**0.5
            if model.vision_tower.config.standardize:
                raise ValueError("Static pooling requires standardize=False")
            kernel = model.vision_tower.config.pooling_kernel_size
            groups = []
            for image, coordinates in enumerate(positions.tolist()):
                width = max(x for x, y in coordinates) + 1
                mapping = {}
                for index, (x, y) in enumerate(coordinates):
                    group = x // kernel + (width // kernel) * (y // kernel)
                    mapping.setdefault(group, []).append(
                        image * positions.shape[1] + index
                    )
                for group in sorted(mapping):
                    if len(mapping[group]) != kernel * kernel:
                        raise ValueError("Incomplete pooling group")
                    groups.append(mapping[group])
            self.register_buffer("indices", torch.tensor(groups, dtype=torch.long))

        def forward(self, hidden_states):
            patches = hidden_states.reshape(-1, hidden_states.shape[-1])[self.indices]
            pooled = patches.float().mean(dim=1) * self.scale
            return self.project(pooled.to(torch.bfloat16)).float()

    class FixedVision(nn.Module):
        def __init__(self):
            super().__init__()
            self.tower, self.project = model.vision_tower, model.embed_vision
            self.register_buffer("positions", positions)
            self.register_buffer("attention_mask", None)
            self.register_buffer("padding", (positions == -1).all(-1))
            self.pool = Pool()

        def forward(self, pixel_values):
            x = self.tower.patch_embedder(pixel_values, self.positions, self.padding)
            rope = self.tower.encoder.rotary_emb(x, self.positions)
            for layer in self.tower.encoder.layers:
                x = layer(
                    x,
                    attention_mask=self.attention_mask,
                    position_embeddings=rope,
                    position_ids=self.positions,
                )
            return self.pool(x)

    return FixedVision().eval().requires_grad_(False)
