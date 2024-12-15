import torch
from training.parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
from rubyinserter import add_ruby

device = "cuda:0" #if torch.cuda.is_available() else "cpu"

# model = ParlerTTSForConditionalGeneration.from_pretrained("Atotti/japanese-parler-tts-mini-bate-finetune-jsut-corpus-625").to(device)
# tokenizer = AutoTokenizer.from_pretrained("Atotti/japanese-parler-tts-mini-bate-finetune-jsut-corpus-625")

# prompt = "今日の天気は？お父さんに電話をかけて"
# description = "Tomoko speaks slightly high-pitched voice delivers her words at a moderate speed with a quite monotone tone in a confined environment, resulting in a quite clear audio recording."
model = None
tokenizer = None

def gen(model_name:str, prompt: str, description: str, output_file_path: str) -> None:
    global model, tokenizer
    if model is None:
        model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = add_ruby(prompt)
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    sf.write(output_file_path, audio_arr, model.config.sampling_rate)
