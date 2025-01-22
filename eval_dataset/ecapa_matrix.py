from speechbrain.inference.speaker import SpeakerRecognition
import os
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
from datasets import load_dataset
import torch

def confusion_matrix(target="", data_num=10, dataset=None):
    # モデルの読み込み https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
    verification = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": "cuda"},
    )

    if dataset:
        # Hugging Face データセットから音声を取得
        ds = load_dataset(dataset, split="train")
        audio_samples = ds[:data_num]  # 指定された数だけ取得
        output = [(sample['array'], f"Sample_{i}") for i, sample in enumerate(audio_samples["audio"])]
    else:
        # ローカルのファイルシステムから音声を取得
        audios_dir = os.path.join("dataset_raw", "audios" if target == "" else f"audios_{target}")
        files = os.listdir(audios_dir)
        output = [os.path.join(audios_dir, x) for x in files][:data_num]

    result = pd.DataFrame()
    if dataset:
        combinations = list(itertools.product(output, repeat=2))
        count = 0
        for combination in combinations:
            # Hugging Face データセットの場合、音声データ（array）を直接使用
            score, prediction = verification.verify_batch(
                torch.tensor(combination[0][0]), torch.tensor(combination[1][0])
            )
            result.loc[combination[0][1], combination[1][1]] = score.item()

            count += 1
            print(f"Count: {count}")
    else:
        combinations = list(itertools.product(output, repeat=2))
        count = 0
        for combination in combinations:
            score, prediction = verification.verify_files(combination[0], combination[1])
            result.loc[
                combination[0].split("/")[-1].replace(".wav", ""),
                combination[1].split("/")[-1].replace(".wav", ""),
            ] = score.item()

            count += 1
            print(f"Count: {count}")

    # 同じファイル同士の組み合わせを除外したデータフレーム
    filtered_result = result.copy()

    # 対角要素（同じファイルのペア）を NaN に設定
    for file in result.index:
        filtered_result.loc[file, file] = float('nan')

    # スコア用のヒートマップ
    plt.figure(figsize=(12, 8))
    plt.title("Score Matrix" if target == "" else f"Score Matrix ({target})")
    sns.heatmap(result, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.1, linecolor='gray', annot_kws={"size": 10}, vmax=1.0, vmin=0.0)
    plt.xticks(rotation=45)
    plt.savefig(os.path.join("eval_dataset_results", f"score_matrix_{target}.png"))

    # ヒストグラムの作成
    # スコア行列の値を1次元配列に変換（NaNを除外）
    score_values = filtered_result.values.flatten()
    score_values = score_values[~np.isnan(score_values)]  # NaN を除外

    # ヒストグラムのプロット
    plt.figure(figsize=(10, 6))
    plt.title("Score Distribution" if target == "" else f"Score Distribution ({target})")
    plt.hist(score_values, bins=30, color="skyblue", edgecolor="gray", alpha=0.7)
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join("eval_dataset_results", f"score_histogram_{target}.png"))

    # 基本統計量の計算
    statistics = {
        "Mean": np.mean(score_values),
        "Median": np.median(score_values),
        "Standard Deviation": np.std(score_values),
        "Minimum": np.min(score_values),
        "Maximum": np.max(score_values),
        "25th Percentile (Q1)": np.percentile(score_values, 25),
        "75th Percentile (Q3)": np.percentile(score_values, 75),
    }

    # 統計量をテキストファイルに保存
    output_dir = "eval_dataset_results"
    os.makedirs(output_dir, exist_ok=True)  # ディレクトリが存在しない場合は作成

    output_file = os.path.join(output_dir, f"statistics_{target}.txt")
    with open(output_file, "w") as f:
        f.write("Score Statistics\n")
        f.write("=================\n")
        for stat_name, value in statistics.items():
            f.write(f"{stat_name}: {value:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="")
    parser.add_argument("--data_num", type=int, default=10)
    parser.add_argument("--dataset", type=str, default=None, help="Hugging Face dataset name")
    args = parser.parse_args()

    confusion_matrix(args.target, args.data_num, args.dataset)