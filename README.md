# GENERATE-JA-TTS

## 現状
JSUTコーパスのBASIC5000というデータでParlerTTSをfine-tuningした。その結果、話者性が付与された音声が生成できる事が確認できた。そのため、ParlerTTSを特定話者の発話データでfine-tuningすることで話者性を付与することができることがわかった。そこで、ゼミのテーマとして私たちの話者性を付与したTTSモデルを開発することを一つの目標とした。

## Road Map
- [ ] Step 1. 読み上げテキストの選択・データ数の検証
- [ ] Step 2. データセットの作成
- [ ] Step 3. fine-tuning of TTS model
- [ ] Step 4. 評価
- [ ] Step 5. 発表準備


### Step 1. 読み上げテキストの選択・データ数の検証
- ParlerTTSがどの程度のデータ量で話者性を付与するのに十分な学習ができるかどうか調査する
  - 現実的に収集可能な発話データ数は、一人あがり100発話程度らしい
  - 既存の設定では学習に失敗した場合、エポック数や学習率などについても調査した上で、今後の方針を検討する
- JVCコーパスの`voiceactress100`が量的にも音素バランス的にも良いかもしれない
  - JSUTコーパスには、`voiceactress100`の音声が含まれているので、検証段階ではJSUTコーパスの`voiceactress100`のが最適だと考えている
    - **JSUTコーパスの`voiceactress100`の注釈付けを行い、データセットを作成し、fine-tuningを行う**

### Step 2. データセットの作成
- **テキストを読み上げて、音声データセットを構築する**
  - テキストと音声の正確な対応関係が重要
  - モデルの精度はデータに依存するので、データセットの質が重要だと考えている
    - 都立大の無響室が使えるなら、そこで録音するのが一番良いと考えている
    - 発音やアクセント、滑舌などはモデルの品質に影響を与えると考えている
- **データセットから注釈付けデータを生成する**
  - Data-Speechを用いて、ラベル付けを行う

### Step 3. fine-tuning of TTS model
- **TTSモデルをfine-tuningする**
  - とりあえず`n`エポックみたいな感じではなく、ちゃんとやる

### Step 4. 評価
- **fine-tuningしたモデルを評価する**
  - 生成された音声の品質を評価する
  - 生成された音声の話者性を評価する
  - 生成された音声の自然さを評価する
  - Zero-shotでの手法との比較を行う 等が考えられる

### Step 5. 発表準備
- **結果をまとめ、発表資料を作成する**

## Documents

### 環境構築
ローカル環境では[Docker](https://docs.docker.jp/get-started/overview.html)を使う。Google Colabでの環境は考え中。

#### build
```bash
docker-compose build
```
GPUが利用できるか確認
```bash
docker-compose run --rm gen-ja-tts bash -c "nvidia-smi"
```

#### コンテナ
シェルに入っちゃうのが楽です。以降のコマンドは、コンテナ内で実行してください。
```bash
docker-compose run --rm gen-ja-tts bash
export HF_TOKEN="your_token"
```
シェルから出るには`exit`を実行してください。`--rm`オプションをつけているので、シェルから抜けた時にコンテナは終了します。

### データセットの作成
HuggingFaceにアップロードするので環境変数にトークンを設定する必要があります。
```bash
uv run main.py format --input_file data/voiceactress100.csv --export_name Atotti/jsut-voiceactress100-datasets
```

### 注釈付けの実行
wip
