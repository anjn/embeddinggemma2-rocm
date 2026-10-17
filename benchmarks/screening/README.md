Configuration screening at budget 70, batch 4; median ms/image over two reversed rounds. PyTorch screening used FP32 weights with BF16 autocast (mixed), with eager text attention; the main matrix uses BF16 parameters. llama.cpp uses BF16 GGUF and FP16 K/V.

| Configuration | ms/image |
|---|---:|
| llama_hip_off | 129.07 |
| llama_hip_on | 111.59 |
| llama_vulkan_off | 149.51 |
| llama_vulkan_on | 137.91 |
| torch_eager | 116.96 |
| torch_sdpa | 92.25 |

Only this representative cell was screened. The main matrix records its selected settings; no per-cell global optimum is claimed. Larger-budget screening with mismatched token grids was rejected and is not used here.
