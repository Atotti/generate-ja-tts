from speechbrain.inference.speaker import SpeakerRecognition
import shutil
from play_model import gen
# モデルの読み込み https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": "cuda"},
)

A = "ONOMATOPEE300_169.wav"
B = "parler_tts_japanese_out.wav"

max_score = 0
while True:
    # 合成音生成
    gen()

    # スコア計算
    score, prediction = verification.verify_files(A, B)

    if score > max_score:
        max_score = score
        print(f"BEST ========== score: {score}, prediction: {prediction} ==========")

        # 合成音を保存

        shutil.copy("parler_tts_japanese_out.wav", "best_score.wav")
    else:
        print(f"score: {score}, prediction: {prediction}")

