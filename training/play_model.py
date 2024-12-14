import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
from rubyinserter import add_ruby

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("Atotti/japanese-parler-tts-mini-bate-finetune-jsut-corpus-625").to(device)
tokenizer = AutoTokenizer.from_pretrained("Atotti/japanese-parler-tts-mini-bate-finetune-jsut-corpus-625")

prompt = "今日の天気は？お父さんに電話をかけて"
description = "Tomoko speaks slightly high-pitched voice delivers her words at a moderate speed with a quite monotone tone in a confined environment, resulting in a quite clear audio recording."


prompt = add_ruby(prompt)

def gen():
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    sf.write("parler_tts_japanese_out.wav", audio_arr, model.config.sampling_rate)

if __name__ == "__main__":
    gen()

