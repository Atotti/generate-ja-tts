# GENERATE-JA-TTS

## 現状
JSUTコーパスのBASIC5000というデータでParlerTTSをfine-tuningした。その結果、話者性が付与された音声が生成できる事が確認できた。そのため、ParlerTTSを特定話者の発話データでfine-tuningすることで話者性を付与することができることがわかった。そこで、ゼミのテーマとして私たちの話者性を付与したTTSモデルを開発することを一つの目標とした。

## Road Map
- [x] Step 1. 読み上げテキストの選択・データ数の検証
- [x] Step 2. データセットの作成
- [ ] Step 3. fine-tuning of TTS model
- [ ] Step 4. 評価
- [ ] Step 5. 発表準備


### Step 1. 読み上げテキストの選択・データ数の検証
- ParlerTTSがどの程度のデータ量で話者性を付与するのに十分な学習ができるかどうか調査する
  - 現実的に収集可能な発話データ数は、一人あがり100発話程度らしい
  - 既存の設定では学習に失敗した場合、エポック数や学習率などについても調査した上で、今後の方針を検討する
- JSUTコーパスの`voiceactress100`が量的にも音素バランス的にも良いかもしれない
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

---

# Documents

## 環境構築
ローカル環境では[Docker](https://docs.docker.jp/get-started/overview.html)を使う。おそらく[NVIDIA container toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)が必要になる。

Google Colabでの実行方法は`generate_ja_tts.ipynb`を参照。Google Colabでも学習が行える事を確認した。むしろColabの方が環境構築も楽だしGPUも強いから学習も速いしで良いかもしれない。
Colabにおいても`uv`を使うことで、python versionに付随するライブラリの互換性の問題を回避している。[Colabノートブック](https://colab.research.google.com/drive/1HzWzgvy5vv417i_jzzOBbpeDxZgdO0cv?usp=sharing)も共有しておく。

### build
```bash
docker-compose build
```
GPUが利用できるか確認
```bash
docker-compose run --rm gen-ja-tts bash -c "uv run main.py cuda-check"
```

### コンテナ
**VSCodeをDockerコンテナにアタッチするのが一番楽です。WSLにアタッチするのと同じ感じでDockerコンテナにアタッチできます。**
1. 何かしらの方法でコンテナを起動 (`sleep INF`になってるのでどんな起動方法でも多分大丈夫)
```bash
docker-compose run --rm gen-ja-tts bash
```
2. VSCode > Remote Explorer > PREMOTE EXPORER を Dev Containers に変更 > gen-ja-tts > Attach current window
3. 環境変数を設定
```bash
export HF_TOKEN="your_token"
```
以降のコマンドは、基本コンテナ内で実行してください。

#### 補足 アタッチしない場合
```bash
docker-compose run --rm gen-ja-tts bash -c "任意のコマンド"
```

※ 毎回`HF_TOKEN`設定するの面倒だから良い感じの方法考え中

※ `.venv`もvolumeしているから、build毎に依存関係をインストールしなおす必要はないと思うのだが、なんか上手くいってなさそう。

## データセットの作成
テキストと音声のデータをデータセットの形式に変換する。HuggingFaceにアップロードするので環境変数にWrite権限のあるアクセストークンを設定する必要があります。
`Atotti/jsut-voiceactress100-datasets`にデータセットをアップロードする場合
```bash
uv run main.py format --input_file data/voiceactress100.csv --export_name Atotti/jsut-voiceactress100-datasets
```

## 注釈付けの実行
注釈付けは3ステップに分かれている。
1. タグ付け
2. 連続変数を離散的なキーワードにマッピング
3. LLMを用いた説明文の生成

内容の詳細は[Natural language guidance of high-fidelity text-to-speech with synthetic annotations](https://arxiv.org/abs/2402.01912)を参照。実際にはこの論文の実装である[dataspeech](https://github.com/huggingface/dataspeech)を利用している。

### タグ付け
`Atotti/jsut-voiceactress100-datasets`のデータセットにタグ付けを行い、`Atotti/jsut-voiceactress100-tags`にアップロードする場合
```bash
uv run data_speech.py "Atotti/jsut-voiceactress100-datasets" --configuration "default" --audio_column_name "audio" --text_column_name "text" --cpu_num_workers 8 --repo_id "jsut-voiceactress100-tags" --apply_squim_quality_estimation
```

### 連続変数を離散的なキーワードにマッピング
`Atotti/jsut-voiceactress100-tags`のデータセットに連続変数を離散的なキーワードにマッピングし、`Atotti/jsut-voiceactress100-keywords`にアップロードする場合
```bash
uv run ./scripts/metadata_to_text.py "Atotti/jsut-voiceactress100-tags" --repo_id "jsut-voiceactress100-keywords" --configuration "default" --cpu_num_workers "8" --avoid_pitch_computation --apply_squim_quality_estimation
```

### LLMを用いた説明文の生成
[gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it)をbfloat16で実行し、`Atotti/jsut-voiceactress100-keywords`のデータセットを用いて`--speaker_name "Tomoko"`として説明文を生成し、`Atotti/jsut-voiceactress100-descriptions`にアップロードする場合
```bash
uv run ./scripts/run_prompt_creation.py --speaker_name "Tomoko" --is_single_speaker --is_new_speaker_prompt --dataset_name "Atotti/jsut-voiceactress100-keywords" --dataset_config_name "default" --model_name_or_path "google/gemma-2-2b-it" --per_device_eval_batch_size 4 --attn_implementation "sdpa" --dataloader_num_workers 4 --torch_dtype "bfloat16" --load_in_4bit --push_to_hub --hub_dataset_id "jsut-voiceactress100-descriptions" --preprocessing_num_workers 4 --output_dir "./output"
```

## モデルの学習
学習データ`Atotti/jsut-voiceactress100-datasets`とメタデータ`Atotti/jsut-voiceactress100-descriptions`を用いてモデルを学習する。学習済みモデル`2121-8/japanese-parler-tts-mini`をfine-tuningする場合。(パラメータは参考実装から変更していない。)

wandbを使うなら、事前にログインしておく。
```bash
uv run wandb login
```

学習の実行
```bash
sh train.sh
```

## 合成音生成
### 一つの音声を生成する場合
```bash
sh gen.sh
```

### ecapaスコアが高い音声を生成する場合
```bash
uv run main.py generate-high-score
```

## 混合行列の作成
`audios/`にある音声ファイルを用いて混合行列を作成する。
```bash
uv run main.py plot-ecapa-confusion-matrix
```
