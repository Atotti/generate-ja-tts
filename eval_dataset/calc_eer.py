import itertools
from speechbrain.inference.speaker import SpeakerRecognition
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

def calc_eer():
    # モデルの読み込み https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
    verification = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": "cuda"},
    )

    # キャッシュファイルの定義
    cache_file = os.path.join("eval_dataset_results", "verification_cache.pkl")

    # キャッシュのロードまたは初期化
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            verification_cache = pickle.load(f)
    else:
        verification_cache = {}

    # ローカルのファイルシステムから音声を取得
    files = {"ayuto": [], "hiroki": [], "olimov": [], "hinako": [], "mako": []}
    for target in files:
        audios_dir = os.path.join("dataset_raw", f"audios_{target}")
        ls = os.listdir(audios_dir)
        files[target] = [os.path.join(audios_dir, x) for x in ls]

    # ペアのリストを作成
    positive_pairs = []
    negative_pairs = []

    for speaker, audio_files in files.items():
        # 同一話者間のペア（Positive）
        positive_pairs.extend(itertools.combinations(audio_files, 2))

        # 異なる話者間のペア（Negative）
        other_speakers = [f for s, f in files.items() if s != speaker]
        for other_files in other_speakers:
            for file_a in audio_files:
                for file_b in other_files:
                    negative_pairs.append((file_a, file_b))

    # スコアとラベルを収集
    scores = []
    labels = []

    def get_score(file_a, file_b):
        pair_key = (file_a, file_b)
        if pair_key in verification_cache:
            return verification_cache[pair_key]
        score, _ = verification.verify_files(file_a, file_b)
        verification_cache[pair_key] = score.item()
        return score.item()

    # Positiveペアの評価
    print("Processing positive pairs...")
    for file_a, file_b in tqdm(positive_pairs, desc="Positive Pairs"):
        scores.append(get_score(file_a, file_b))
        labels.append(1)  # 同一話者は1

    # Negativeペアの評価
    print("Processing negative pairs...")
    for file_a, file_b in tqdm(negative_pairs, desc="Negative Pairs"):
        scores.append(get_score(file_a, file_b))
        labels.append(0)  # 異なる話者は0

    # キャッシュの保存
    with open(cache_file, "wb") as f:
        pickle.dump(verification_cache, f)

    # FAR, FRR, EERの計算
    thresholds = np.linspace(min(scores), max(scores), 1000)
    fpr = []  # False Acceptance Rate
    fnr = []  # False Rejection Rate

    for threshold in thresholds:
        false_accepts = sum((s > threshold and l == 0) for s, l in zip(scores, labels))
        false_rejects = sum((s <= threshold and l == 1) for s, l in zip(scores, labels))
        true_accepts = sum((s > threshold and l == 1) for s, l in zip(scores, labels))
        true_rejects = sum((s <= threshold and l == 0) for s, l in zip(scores, labels))

        total_negatives = false_accepts + true_rejects
        total_positives = false_rejects + true_accepts

        fpr.append(false_accepts / total_negatives if total_negatives > 0 else 0)
        fnr.append(false_rejects / total_positives if total_positives > 0 else 0)

    fpr = np.array(fpr)
    fnr = np.array(fnr)

    # EER計算
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[eer_index]
    eer_threshold = thresholds[eer_index]

    print(f"Equal Error Rate (EER): {eer * 100:.2f}%")
    print(f"EER Threshold: {eer_threshold}")

    # FARとFRRのプロット
    plt.figure()
    plt.plot(thresholds, fpr, label="FAR (False Acceptance Rate)")
    plt.plot(thresholds, fnr, label="FRR (False Rejection Rate)")
    plt.axvline(eer_threshold, color="red", linestyle="--", label=f"EER = {eer * 100:.2f}% at Threshold = {eer_threshold:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.title("FAR and FRR")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join("eval_dataset_results", "eer.png"))

if __name__ == "__main__":
    calc_eer()
