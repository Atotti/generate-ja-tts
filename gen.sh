#!bin/sh

time_stamp=$(date +%Y%m%d%H%M%S)
file_name="generated_sound/${time_stamp}.wav"

uv run main.py generate \
    --model_name "Atotti/japanese-parler-tts-mini-jsut-voiceactress100" \
    --prompt "今日の天気は？お父さんに電話をかけて" \
    --description "Tomoko's words are delivered in a very monotone voice, with very poor recording, as the sounds are muffled by a very close-sounding environment. The pace of her speech is slow." \
    --output_file_path "$file_name"
