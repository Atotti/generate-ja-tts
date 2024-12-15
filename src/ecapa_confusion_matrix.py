from speechbrain.inference.speaker import SpeakerRecognition
import os
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def confusion_matrix():
    # モデルの読み込み https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
    verification = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": "cuda"},
    )

    output = os.listdir("audios")
    output = [os.path.join("audios", x) for x in output]
    result = pd.DataFrame()
    combinations = list(itertools.product(output, repeat=2))

    for combination in combinations:
        score, prediction = verification.verify_files(combination[0], combination[1])
        result.loc[combination[0].split("/")[-1].replace(".wav", ""), combination[1].split("/")[-1].replace(".wav", "")] = score.item()

    # スコア用のヒートマップ
    plt.figure(figsize=(12, 8))
    plt.title("Score Matrix")
    sns.heatmap(result, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.1, linecolor='gray', annot_kws={"size": 10})
    plt.xticks(rotation=45)
    plt.savefig("score_matrix.png")

