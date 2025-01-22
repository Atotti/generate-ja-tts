import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os
from datasets import load_dataset

def analyze_dataset(dataset_name, target=""):
    # Hugging Faceデータセットの読み取り
    dataset = load_dataset(dataset_name, split="train")

    # 必要な情報を抽出
    data = [
        {
            'text': item['text'],
            'snr': item['snr'],
            'pesq': item['pesq']
        }
        for item in dataset
    ]

    # DataFrameに変換
    df = pd.DataFrame(data)

    # 平均値の計算
    snr_mean = df['snr'].mean()
    pesq_mean = df['pesq'].mean()

    # 結果を表示
    print(f"Average SNR: {snr_mean:.2f} dB")
    print(f"Average PESQ: {pesq_mean:.2f}")

    # 保存用ディレクトリ作成
    output_dir = "eval_dataset_results"
    os.makedirs(output_dir, exist_ok=True)

    # ヒストグラムを描画して保存
    def plot_and_save_histogram(column, title, xlabel, filename):
        plt.figure(figsize=(10, 6))
        plt.hist(df[column], bins=10, alpha=0.75, edgecolor='black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path)
        plt.close()

    # SNRのヒストグラム
    plot_and_save_histogram('snr', "SNR Distribution" if target == "" else f"SNR Distribution ({target})", "SNR (dB)", f"snr_histogram_{target}.png")

    # PESQのヒストグラム
    plot_and_save_histogram('pesq', "PESQ Distribution" if target == "" else f"PESQ Distribution ({target})", "PESQ Score", f"pesq_histogram_{target}.png")

    # 基本統計量を計算して保存
    statistics = {
        "Average SNR (dB)": snr_mean,
        "Average PESQ": pesq_mean,
        "SNR Std Dev": df['snr'].std(),
        "PESQ Std Dev": df['pesq'].std(),
        "SNR Min": df['snr'].min(),
        "SNR Max": df['snr'].max(),
        "PESQ Min": df['pesq'].min(),
        "PESQ Max": df['pesq'].max()
    }

    stats_file = os.path.join(output_dir, f"analyze_snr_pesq_{target}.txt")
    with open(stats_file, "w") as f:
        for stat_name, value in statistics.items():
            f.write(f"{stat_name}: {value:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze SNR and PESQ from a Hugging Face dataset.")
    parser.add_argument("dataset_name", type=str, help="The name of the Hugging Face dataset to analyze.")
    parser.add_argument("--target", type=str, default="")
    args = parser.parse_args()

    analyze_dataset(args.dataset_name, args.target)
