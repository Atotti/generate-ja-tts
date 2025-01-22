import itertools
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from speechbrain.inference.speaker import SpeakerRecognition
import torch

def calc_eer():
    # モデルの読み込み https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
    verification = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": "cuda"},
    )

    # キャッシュファイルの定義
    embeddings_cache_file = os.path.join("eval_dataset_results", "embeddings_cache.pkl")

    # キャッシュのロードまたは初期化
    if os.path.exists(embeddings_cache_file):
        with open(embeddings_cache_file, "rb") as f:
            embeddings_cache = pickle.load(f)
    else:
        embeddings_cache = {}

    # ローカルのファイルシステムから音声を取得
    files = {"ayuto": [], "hiroki": [], "olimov": [], "hinako": [], "mako": []}
    for target in files:
        audios_dir = os.path.join("dataset_raw", f"audios_{target}")
        ls = os.listdir(audios_dir)
        files[target] = [os.path.join(audios_dir, x) for x in ls]

    generated_audio_files = {"ayuto": [], "hiroki": [], "olimov": [], "hinako": [], "mako": []}
    for target in files:
        audios_dir = os.path.join("dataset_raw", "generated_audios", target)
        ls = os.listdir(audios_dir)
        generated_audio_files[target] = [os.path.join(audios_dir, x) for x in ls]

    # 埋め込みを取得または計算
    def get_embedding(file):
        if file in embeddings_cache:
            return embeddings_cache[file]
        waveform = verification.load_audio(file)
        waveform = waveform.unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")  # GPUに移動
        embedding = verification.encode_batch(waveform).cpu()
        embeddings_cache[file] = embedding
        return embedding

    # ペアのリストを作成
    positive_pairs = []
    negative_pairs = []
    attack_pairs = []

    for speaker, audio_files in files.items():
        # 同一話者間のペア（Positive）
        positive_pairs.extend(itertools.combinations(audio_files, 2))

        # 異なる話者間のペア（Negative）
        other_speakers = [f for s, f in files.items() if s != speaker]
        for other_files in other_speakers:
            for file_a in audio_files:
                for file_b in other_files:
                    negative_pairs.append((file_a, file_b))

    # 対象話者の実発話と合成音声のペア
    for speaker, audio_files in generated_audio_files.items():
        for file_a in audio_files:
            for file_b in files[speaker]:
                attack_pairs.append((file_a, file_b))

    # スコアとラベルを収集
    scores = []
    labels = []

    def calculate_cosine_similarity(file_a, file_b):
        embedding_a = get_embedding(file_a)
        embedding_b = get_embedding(file_b)
        score = verification.similarity(embedding_a, embedding_b)
        return score.item()

    # Positiveペアの評価
    print("Processing positive pairs...")
    for file_a, file_b in tqdm(positive_pairs, desc="Positive Pairs"):
        scores.append(calculate_cosine_similarity(file_a, file_b))
        labels.append(1)  # 同一話者は1

    # Negativeペアの評価
    print("Processing negative pairs...")
    for file_a, file_b in tqdm(negative_pairs, desc="Negative Pairs"):
        scores.append(calculate_cosine_similarity(file_a, file_b))
        labels.append(0)  # 異なる話者は0
    # Attackペアの評価
    print("Processing attacking pairs...")
    for file_a, file_b in tqdm(attack_pairs, desc="Attack Pairs"):
        scores.append(calculate_cosine_similarity(file_a, file_b))
        labels.append(-1)  # 合成音声とのペアは-1

    # 埋め込みキャッシュの保存
    with open(embeddings_cache_file, "wb") as f:
        pickle.dump(embeddings_cache, f)

    # FAR, FRR, EERの計算
    thresholds = np.linspace(min(scores), max(scores), 1000)
    fpr = []  # False Acceptance Rate
    fnr = []  # False Rejection Rate
    fpr_no_attack = []  # False Acceptance Rate (合成音声を含まない場合)

    # 合成音声を含む場合
    print("Calculating EER...")
    for threshold in tqdm(thresholds, desc="Thresholds"):
        false_accepts = sum((s > threshold and l <= 0 ) for s, l in zip(scores, labels))
        false_rejects = sum((s <= threshold and l == 1) for s, l in zip(scores, labels))
        true_accepts = sum((s > threshold and l == 1) for s, l in zip(scores, labels))
        true_rejects = sum((s <= threshold and l <= 0) for s, l in zip(scores, labels))

        total_negatives = false_accepts + true_rejects
        total_positives = false_rejects + true_accepts

        fpr.append(false_accepts / total_negatives if total_negatives > 0 else 0)
        fnr.append(false_rejects / total_positives if total_positives > 0 else 0)

    # 合成音声を含まない場合
    print("Calculating EER (without attack)...")
    for threshold in tqdm(thresholds, desc="Thresholds (without attack)"):
        false_accepts = sum((s > threshold and l == 0) for s, l in zip(scores, labels) if l != -1)
        false_rejects = sum((s <= threshold and l == 1) for s, l in zip(scores, labels) if l != -1)
        true_accepts = sum((s > threshold and l == 1) for s, l in zip(scores, labels) if l != -1)
        true_rejects = sum((s <= threshold and l == 0) for s, l in zip(scores, labels) if l != -1)

        total_negatives = false_accepts + true_rejects
        total_positives = false_rejects + true_accepts

        fpr_no_attack.append(false_accepts / total_negatives if total_negatives > 0 else 0)

    fpr = np.array(fpr)
    fnr = np.array(fnr)
    fpr_no_attack = np.array(fpr_no_attack)

    # EER計算
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[eer_index]
    eer_threshold = thresholds[eer_index]

    print(f"Equal Error Rate (EER): {eer * 100:.2f}%")
    print(f"EER Threshold: {eer_threshold}")

    # FARとFRRのプロット
    plt.figure()
    plt.plot(thresholds, fpr, label="FAR (False Acceptance Rate)", color="#F24343")
    plt.plot(thresholds, fnr, label="FRR (False Rejection Rate)", color="#F2AC43")
    plt.plot(thresholds, fpr_no_attack, label="FAR (without attack)", color="#4285F4")
    # plt.axvline(eer_threshold, color="red", linestyle="--", label=f"EER = {eer * 100:.2f}% at Threshold = {eer_threshold:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("Error Rate")
    plt.title("FAR / FRR")
    # plt.legend()
    plt.grid()
    plt.savefig(os.path.join("eval_dataset_results", "eer_with_embedding.png"))

if __name__ == "__main__":
    calc_eer()
