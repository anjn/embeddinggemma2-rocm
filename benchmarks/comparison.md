| Token budget | Batch | Native | PyTorch BF16 | llama.cpp HIP | llama.cpp Vulkan |
|---:|---:|---:|---:|---:|---:|
| 70 | 1 | 57.64 | 100.74 | 123.68 | 179.30 |
| 70 | 4 | 50.67 | 83.73 | 111.31 | 138.23 |
| 70 | 32 | 27.23 | 75.30 | 104.51 | 125.78 |
| 70 | 128 | 26.71 | 74.62 | excluded | 132.21 |
| 140 | 1 | 102.50 | 189.69 | 234.46 | 296.23 |
| 140 | 4 | 92.92 | 160.55 | 204.04 | 242.52 |
| 140 | 32 | 82.43 | 151.71 | 206.86 | 251.55 |
| 140 | 128 | 82.48 | 152.30 | excluded | 257.52 |
| 280 | 1 | 241.38 | 370.66 | 576.94 | 636.43 |
| 280 | 4 | 205.52 | 347.39 | 550.68 | 631.52 |
| 280 | 32 | 202.44 | 338.81 | 603.91 | 619.61 |
| 280 | 128 | failed (139) | omitted | excluded | omitted |

All numeric cells are median ms/image across two reversed-order runs. Failed, unsupported and unmeasured cases remain explicit.
