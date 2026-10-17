# 速度比較の再現手順

同じGPUを使い、他のコンパイル・推論を停止して測定します。ドライバー、電力設定、CPUスレッド数、ライブラリのリビジョン、精度、Attention設定を固定して記録します。速度測定には固定シードで生成した512×288のRGB配列を使うため、非公開の写真データは不要です。検索精度は別の実データで検証します。

`eg2 benchmark`は`native`、`torch`、`llama`を選べます。時間と状態をJSON、埋め込みをNPYへ保存します。比較する条件はbudget 70/140/280とbatch 1/4/32/128です。実際の画像トークン数、系列長、内部バッチを結果に記録します。内部で分割した場合も、要求全体の処理時間を測定します。

llama.cppは`tools/build_llama.sh`で固定リビジョンと直接呼び出し用アダプターをビルドします。対応するBF16モデルとマルチモーダルプロジェクターのGGUFを指定してください。HIP/Vulkan、Flash Attention、内部バッチを記録します。未測定の構成を最速と記載しないでください。

ランナーのJSON設定例は[英語版](benchmarks.md)を参照してください。

```sh
python benchmarks/run_matrix.py --config runners.json --output benchmarks/results/my-gpu
python benchmarks/render.py benchmarks/results/my-gpu --output benchmarks/comparison.md
```

各バックエンドを別プロセスで実行し、順序を反転した2ラウンドを測定します。各ラウンドで3回以上のウォームアップと5回以上の本測定を行います。表の数値は両ラウンドの全測定値の中央値で、単位はms/画像です。片方だけの成功や、入力・精度・バージョンが異なる結果は集計しません。未測定と未対応を区別します。速度を公開する前に、出力の正しさを別途確認してください。

`tools/Dockerfile.baselines`で、HIP BLASヘッダーとVulkanのビルド依存を追加したイメージを作れます。llama.cppの固定コミットとGGUFの取得元・リビジョンは[英語版](benchmarks.md)に記載しています。モデルファイルは別途取得し、Gitには含めません。

PyTorch側はBF16パラメーター、VisionのSDPA、テキスト側のeager attentionを既定値とします。今回のROCm環境では、テキスト側のSDPAはFP32でも元のFP32モデルとの一致検証に失敗したため、測定に使いません。`--torch-precision mixed`はFP32パラメーターとBF16 autocast、`--torch-precision fp32`はFP32です。精度設定を変えた結果は別の比較対象として扱います。

512×288入力で画像トークン数をそろえるため、llama.cppには最小値と最大値を57/70、111/140、249/280として渡します。実際の画像トークン数は66/120/264、全系列長は70/124/268になります。上限だけを指定すると、大きいbudgetでも最小グリッドのまま処理されます。アダプターは実トークン数を確認し、一致しない条件を未対応として記録します。llama.cppは元のバイキュービック補間とパディングを使うため、トークン数が同じでも画素テンソルまで同一ではありません。

[設定例](../benchmarks/runners.example.json)にはllama.cppの両バックエンドを含めています。HIPのbatch 128を除外する設定は、gfx1150での検証不合格を反映しています。別の環境では検証後に見直してください。`skip_batches`を指定すると、`skip_reason`の理由とともに`validation_failed`を記録し、速度値やAPI未対応として扱いません。[設定比較の結果](../benchmarks/screening/README.md)にはAttention設定の測定値と各ラウンドのJSONを保存しています。

速度測定の入力はNumPyのseed 812で16画像を生成し、batch 32・128ではそれらを繰り返します。バックエンド間で同一のバイト列を使用します。検索精度と入力順変更の検証には実画像を別途使用します。

この版では、budget 280の画像要求と準備時の実行バッチを32枚以下に制限します。gfx1150で直接batch 128と32枚ずつの4分割の両方が安定性検証に失敗したため、33枚以上の要求には`EG2_UNSUPPORTED`（Pythonでは`UnsupportedProfile`、HTTPでは503）を返します。32枚以下の要求に分けて送信してください。budget 70・140は128枚まで受け付けます。
