# AMD GPU向けEmbeddingGemma 2

[English](README.md)

`google/embeddinggemma-2`の画像・テキスト埋め込みをAMD GPUで実行するライブラリです。C ABI、C++のRAIIラッパー、Pythonバインディング、HTTPサーバー、GPU別のチューニングツールを提供します。出力はL2正規化済みの768次元FP32ベクトルです。

画像はHIPによる前処理、MIGraphX、LibTorchのAttentionで実行します。テキストは事前に準備したLibTorchモジュールとRust製トークナイザーを使います。C/C++からの推論にはPythonインタープリターを使いません。モデルの準備とHTTPサーバーにはPythonが必要です。

Radeon 890M（gfx1150）には、token budget 70、アスペクト比16:9、Vision内部バッチ16用の高速化処理を用意しています。その他の形状には通常のコンパイル処理を使います。他のAMD GPUは未検証であり、使用するGPUでのコンパイルと検証が必要です。モデルとコンパイル済みデータは別途取得・生成します。


`eg2 share`で、複数の画像・テキストプロファイル間で重みと最大サイズの作業領域を共有できます。HIPチューナーはプロファイルごとの設定を探索し、検証済みのGPU設定を`profiles/gpus/`へ保存できます。一致する環境では探索せずに再利用します。詳しくは[チューニングとメモリ共有](docs/tuning.ja.md)を参照してください。

## MobileCLIP2-S2

MobileCLIP2-S2の画像・テキストを512次元ベクトルに変換するAPIも利用できます。画像はMIGraphX FP16でbatch 1・4、Ryzen AI SW 1.9 BF16でbatch 1に対応します。テキストはCPU FP32で実行します。Pythonでは`MobileCLIP2Embedder`、C/C++では`libmobileclip2`を使います。[準備・ビルド・利用例](docs/mobileclip2.md)を参照してください。検索インデックスはEmbeddingGemmaとは別に作成してください。

## ビルドとモデルの準備

コンテナーを使うにはLinux、対応AMD GPU、Docker、ホスト側のROCmドライバーが必要です。モデルの利用条件を確認し、必要なアクセス権を取得してください。認証が必要な場合は、シェルで`HF_TOKEN`を設定します。

```sh
docker build --build-arg GPU_ARCH=gfx1150 -t embeddinggemma2-rocm:local .
mkdir -p models artifacts
docker run --rm -e HF_TOKEN -v "$PWD/models:/data/models" \
  embeddinggemma2-rocm:local fetch --output /data/models/google

docker run --rm --device=/dev/kfd --device=/dev/dri \
  --group-add "$(stat -c %g /dev/kfd)" \
  -v "$PWD/models:/data/models:ro" -v "$PWD/artifacts:/data/artifacts" \
  embeddinggemma2-rocm:local prepare --model models/google \
  --output artifacts/my-gpu --budgets 70 --batches 1 4

docker run --rm --device=/dev/kfd --device=/dev/dri \
  --group-add "$(stat -c %g /dev/kfd)" \
  -v "$PWD/models:/data/models:ro" -v "$PWD/artifacts:/data/artifacts" \
  embeddinggemma2-rocm:local prepare-text --model models/google \
  --output artifacts/my-gpu --batches 1 4 --lengths 32 128

docker run --rm -v "$PWD/artifacts:/data/artifacts" \
  embeddinggemma2-rocm:local share --bundle artifacts/my-gpu \
  --output artifacts/my-gpu-shared
```

準備には時間とGPUメモリーを要するため、小さいバッチから始めてください。コンパイル済みデータにはモデルの重みを含み、GPUとソフトウェア環境に依存します。gfx1150のbudget 70、batch 32/128ではVision内部バッチ16を選び、専用の高速化を試みます。グラフ構造の検査に失敗した場合は、ログに理由を記録し、通常のコンパイル結果を残します。

別のアスペクト比は`--aspect WIDTH HEIGHT`で準備します。未準備の比率はエラーになります。暗黙の16:9クロップは行いません。準備後は、必要な依存ライブラリとコンパイル済みデータがあればオフラインで実行できます。

[ソースビルド](docs/build.ja.md)、[API](docs/api.ja.md)、[チューニング](docs/tuning.ja.md)に詳細があります。

## アプリケーションへの組み込み

```python
import numpy as np
from embeddinggemma2_rocm import Embedder

with Embedder("artifacts/my-gpu-shared") as model:
    images = np.zeros((4, 288, 512, 3), dtype=np.uint8)
    image_vectors = model.encode_images(images, token_budget=70)
    text_vectors = model.encode_texts(["猫", "街並み"], prompt="query")
```

`EG2_LIBRARY_PATH`に`libeg2.so`のパスを設定します。同じ呼び出しの画像は縦横サイズを揃えてください。要求バッチが準備済みバッチより大きい場合は分割して実行します。コンテキスト内の呼び出しは直列に処理します。

この版では、budget 280の画像要求と準備時の実行バッチを32枚以下に制限します。gfx1150で直接batch 128と32枚ずつの4分割の両方が安定性検証に失敗したため、33枚以上の要求には`EG2_UNSUPPORTED`（Pythonでは`UnsupportedProfile`、HTTPでは503）を返します。32枚以下の要求に分けて送信してください。budget 70・140は128枚まで受け付けます。

C/C++では[公開ヘッダー](include/eg2/eg2.h)と[使用例](examples/)を参照してください。

## HTTPサーバー

```sh
docker run --rm --device=/dev/kfd --device=/dev/dri \
  --group-add "$(stat -c %g /dev/kfd)" -p 127.0.0.1:8000:8000 \
  -v "$PWD/artifacts:/data/artifacts:ro" embeddinggemma2-rocm:local \
  serve --bundle artifacts/my-gpu-shared --host 0.0.0.0

curl http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"embeddinggemma-2","input":["a cat"],"input_type":"query"}'
```

`/v1/embeddings`はテキストと画像の入力を受け付けます。画像形式はOpenAIのテキスト埋め込みAPIに対する拡張です。[API仕様](docs/api.ja.md)を参照してください。外部公開時の認証はリバースプロキシなどで設定します。

## 速度比較

比較条件は、固定seedで生成した512×288の16:9 RGB入力、token budget 70/140/280、要求batch 1/4/32/128です。対応する組み合わせを測定します。budgetは上限であり、実際の画像トークン数、系列長、内部バッチ、精度、バックエンド設定を測定結果に記録します。

CPU上のRGB配列を受け取ってから、正規化済みFP32埋め込みをCPUへ返すまでを計測します。パッチ前処理とCPU/GPU間転送を含み、ファイルのデコード、事前リサイズ、モデル読み込み、コンパイル、ウォームアップを除きます。ネイティブ側は半ピクセル座標のバイリニア補間と1/255スケーリングを統合し、PyTorch側は元のバイキュービック・アンチエイリアス処理を使います。この前処理の近似も、速度と検索精度の評価対象に含めます。HTTPを介さず、各ライブラリを直接呼び出します。バックエンドごとにプロセスを分け、実行順と逆順の各ラウンドで3回以上のウォームアップと5回以上の測定を行います。

追加のHIP探索で、budget 70・batch 1の**35.74 ms/画像**の[検証済み設定](profiles/gpus/gfx1150/budget70-b1.json)を保存しました。以下の比較表は従来の測定セットを維持し、追加分の[測定記録](benchmarks/results/gfx1150-hip-tuned/)と[検証結果](docs/validation.ja.md)を別に示しています。

<!-- benchmark-table:start -->

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

数値の単位は**ms/画像**で、順序を反転した2ラウンドの中央値です。`excluded`は数値検証不合格、`failed (139)`はネイティブ版の安定性検証失敗、`omitted`は残りの測定を省略した条件または1ラウンドのみ完了した条件です。完了済みの大バッチ比較値は残し、PyTorch・llama.cppのbatch 32・128の追加測定は行いません。

budget 70・batch 128ではネイティブ版が**26.71 ms/画像**となり、このGPUでの処理速度はPyTorch BF16の**2.79倍**、llama.cpp Vulkanの**4.95倍**でした。

<!-- benchmark-table:end -->

測定環境はRadeon 890M（`gfx1150`）、ROCm 7.2、PyTorch 2.14.1+rocm7.2です。PyTorchはBF16パラメーター、VisionのSDPA、テキスト側のeager attentionを使用します。stock llama.cppはBF16 GGUF重み、FP16 K/V、Flash Attention有効とし、HIP版とVulkan版を分けて示します。実際の画像トークン数は**66/120/264**、全系列長は**70/124/268**です。HIPのbatch 128は実画像の入力順変更検査で不合格だったため`validation_failed`として除外し、より大きいbudgetのHIP batch 128も未検証として扱います。[各ラウンドのJSON](benchmarks/results/gfx1150/)、[環境](benchmarks/environment.json)、[設定比較](benchmarks/screening/README.md)、[再現手順](docs/benchmarks.ja.md)を添付しています。

ネイティブ版のbudget 70・batch 128は、固定した**256画像・1280キャプション**全件で検索精度の基準を満たしました。R@1はテキスト→画像が**86.328125%**、画像→テキストが**93.359375%**です。元のCPU FP32の86.25% / 94.140625%に対する低下幅は−0.078125 / 0.78125ポイントで、両方向とも1ポイント以内でした。ほかの形状とGPUは個別の検索精度検証が必要です。[検証の詳細](docs/validation.ja.md)には、比較対象の検索精度、ネイティブ組み込み、汎用アスペクト比、チューニングと採用処理の結果を記載しています。

## ライセンス

本リポジトリの独自コードは[MIT](LICENSE)です。モデルと依存ライブラリには、それぞれの利用条件が適用されます。[依存ライブラリの表示](THIRD_PARTY_NOTICES.md)も参照してください。
