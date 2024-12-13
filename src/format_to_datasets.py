from datasets import DatasetDict, Audio

def format_to_datasets(input_file: str, export_name: str):
    """Format data into datasets

    Args:
        input_file (str): csv file path
        export_name (str): name of the dataset to export to HuggingFace Hub
    """
    # CSVファイルからデータセットを作成
    dataset = DatasetDict.from_csv({"train": input_file})

    # 音声データのカラムをAudio型にキャスト
    dataset = dataset.cast_column("audio", Audio())

    # データセットをHuggingFace Hubにプッシュ（JSUTコーパスの再配布は許可されていないのでプライベートに）
    dataset.push_to_hub(export_name, private=True)
