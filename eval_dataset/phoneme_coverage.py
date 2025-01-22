import pyopenjtalk
import csv
import os

# 簡易音素変換マッピング
kana_to_phoneme = {
    'ア': ['a'], 'イ': ['i'], 'ウ': ['u'], 'エ': ['e'], 'オ': ['o'],
    'カ': ['k', 'a'], 'キ': ['k', 'i'], 'ク': ['k', 'u'], 'ケ': ['k', 'e'], 'コ': ['k', 'o'],
    'サ': ['s', 'a'], 'シ': ['s', 'i'], 'ス': ['s', 'u'], 'セ': ['s', 'e'], 'ソ': ['s', 'o'],
    'タ': ['t', 'a'], 'チ': ['t', 'i'], 'ツ': ['t', 'u'], 'テ': ['t', 'e'], 'ト': ['t', 'o'],
    'ナ': ['n', 'a'], 'ニ': ['n', 'i'], 'ヌ': ['n', 'u'], 'ネ': ['n', 'e'], 'ノ': ['n', 'o'],
    'ハ': ['h', 'a'], 'ヒ': ['h', 'i'], 'フ': ['h', 'u'], 'ヘ': ['h', 'e'], 'ホ': ['h', 'o'],
    'マ': ['m', 'a'], 'ミ': ['m', 'i'], 'ム': ['m', 'u'], 'メ': ['m', 'e'], 'モ': ['m', 'o'],
    'ヤ': ['y', 'a'], 'ユ': ['y', 'u'], 'ヨ': ['y', 'o'],
    'ラ': ['r', 'a'], 'リ': ['r', 'i'], 'ル': ['r', 'u'], 'レ': ['r', 'e'], 'ロ': ['r', 'o'],
    'ワ': ['w', 'a'], 'ヲ': ['o'], 'ン': ['n'],
    'ガ': ['g', 'a'], 'ギ': ['g', 'i'], 'グ': ['g', 'u'], 'ゲ': ['g', 'e'], 'ゴ': ['g', 'o'],
    'ザ': ['z', 'a'], 'ジ': ['z', 'i'], 'ズ': ['z', 'u'], 'ゼ': ['z', 'e'], 'ゾ': ['z', 'o'],
    'ダ': ['d', 'a'], 'ヂ': ['d', 'i'], 'ヅ': ['d', 'u'], 'デ': ['d', 'e'], 'ド': ['d', 'o'],
    'バ': ['b', 'a'], 'ビ': ['b', 'i'], 'ブ': ['b', 'u'], 'ベ': ['b', 'e'], 'ボ': ['b', 'o'],
    'パ': ['p', 'a'], 'ピ': ['p', 'i'], 'プ': ['p', 'u'], 'ペ': ['p', 'e'], 'ポ': ['p', 'o'],
    'ッ': ['q'], 'ー': ['-']  # 促音や長音
}

# 音素セットを定義
phoneme_set = set(['a', 'i', 'u', 'e', 'o', 'k', 'g', 's', 'z', 't', 'd', 'n', 'h', 'b', 'p', 'm', 'y', 'r', 'w', 'ŋ', 'ん', 'っ', 'ー'])

# 日本語テキストを音素列に変換する関数
def japanese_to_phonemes(text):
    kana_text = pyopenjtalk.g2p(text, kana=True)  # ひらがなに変換
    phonemes = []
    for char in kana_text:
        phonemes.extend(kana_to_phoneme.get(char, []))  # 音素に変換
    return phonemes

# CSVファイルを読み込む関数
def read_csv_column_to_list(file_path, column_index):
    text_list = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        # 各行の指定列をリストに追加
        for row in reader:
            if len(row) > column_index:  # 列が存在する場合のみ
                text_list.append(row[column_index])
    return text_list

# CSVファイルのパス
csv_file_path = os.path.join("dataset", "voiceactress100_ex.csv")  # 適宜変更

# 2列目（インデックス1）のデータを取得
text_data_list = read_csv_column_to_list(csv_file_path, column_index=1)

# 全テキストデータを合算して音素を抽出
all_phoneme_data = []
for text in text_data_list:
    all_phoneme_data.extend(japanese_to_phonemes(text))

# 合算された音素データからカバレッジを計算
extracted_phonemes = set(all_phoneme_data)
coverage = len(extracted_phonemes.intersection(phoneme_set)) / len(phoneme_set)

# 結果を表示
print(f"データセットの音素カバレッジ: {coverage * 100:.2f}%")