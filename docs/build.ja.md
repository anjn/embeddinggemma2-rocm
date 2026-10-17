# ソースビルドとオフライン実行

DockerfileではROCm 7.2とROCm版PyTorch 2.14.1を使います。ネイティブ実行にもMIGraphXとLibTorchが必要です。システム側とPyTorch側でHIPライブラリを二重にロードすると終了時に異常終了する場合があるため、CMakeで同じライブラリを参照する別名ディレクトリを作ります。インストール先の構造を維持してください。

ソースビルドにはCMake 3.22以上、C++20、HIP、MIGraphX、Rust/Cargo、ABIの合うROCm版LibTorchが必要です。CMakeは固定版のMIGraphX・JSONヘッダーをビルド先へ取得します。Rustの依存関係はCargo.lockで固定しています。

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install 'torch==2.14.1+rocm7.2' 'torchvision==0.29.1+rocm7.2' \
  --index-url https://download.pytorch.org/whl/rocm7.2 --extra-index-url https://pypi.org/simple
pip install '.[prepare,server,test]'
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DTORCH_ROOT="$VIRTUAL_ENV/lib/python3.12/site-packages/torch" \
  -DEG2_GPU_ARCH=gfx1150 -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build build -j2
cmake --install build
export EG2_LIBRARY_PATH="$PWD/install/lib/libeg2.so"
# gfx1150のネイティブAttentionではAOTritonの実験的対応を有効にする
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

`EG2_CXX11_ABI`はLibTorchの設定に合わせます。Rustを別途ビルドした場合は`EG2_TOKENIZER_LIBRARY`に静的ライブラリを指定できます。オフラインビルドでは`build/deps`へ必要なヘッダーを事前配置します。

`eg2 fetch --output models/google`で固定リビジョンのモデルを取得します。認証が必要な場合はHugging Faceの標準的な認証方法を使ってください。準備コマンドはローカルのモデルディレクトリを受け取ります。準備後の推論にはネットワーク接続を要しません。

GPUアーキテクチャやROCm・MIGraphX・LibTorchを変更した場合は再コンパイルしてください。`eg2 info --bundle artifacts/my-gpu`で準備済みプロファイルを確認できます。`install/bin/eg2-example artifacts/my-gpu 1`はPythonを使わずに動作するC++の確認用プログラムです。

gfx1150では、一部の形状に対応するFP32のMLIR Attentionカーネルがありません。MIGraphXが`No solutions`を返した場合、準備ツールは別プロセスで通常のGPU演算を使って再試行します。この場合もGPUで実行します。MIGraphXがコンパイラーの環境設定をキャッシュするため、再試行には別プロセスが必要です。
