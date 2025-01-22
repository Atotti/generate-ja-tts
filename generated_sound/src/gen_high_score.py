from speechbrain.inference.speaker import SpeakerRecognition
import shutil
from src.play_model import gen

def gen_high_score():
    # モデルの読み込み https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
    verification = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": "cuda"},
    )

    A = "generated_sound/ONOMATOPEE300_169.wav"
    B = "generated_sound/parler_tts_japanese_out.wav"

    max_score = 0
    while True:
        # 合成音生成
        gen(
            model_name="Atotti/japanese-parler-tts-mini-jsut-voiceactress100",
            prompt="今日の天気は？お父さんに電話をかけて",
            description="Tomoko's voice delivers her words at a moderate speed with a quite monotone tone slightly low pitch in a confined environment. The pace of her speech is slow, resulting in a quite clear audio recording.",
            output_file_path=B
            )

        # スコア計算
        score, prediction = verification.verify_files(A, B)

        if score > max_score:
            max_score = score
            print(f"BEST ========== score: {score}, prediction: {prediction} ==========")

            # ベストスコアの合成音を保存
            shutil.copy(B, "generated_sound/best_score.wav")
        else:
            print(f"score: {score}, prediction: {prediction}")

