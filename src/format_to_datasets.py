from datasets import DatasetDict, Audio

def format_to_datasets(input_file: str, export_name: str, name: str):
    """Format data into datasets

    Args:
        input_file (str): csv file path
        export_name (str): name of the dataset to export to HuggingFace Hub
        name (str): name of the dataset
    """
    # CSVファイルからデータセットを作成
    dataset = DatasetDict.from_csv({"train": input_file})

    # csvファイル中の<name>をnameに置換
    dataset = dataset.map(
        lambda x: {
            "audio": x["audio"],
            "text": x["text"].replace("<name>", name)
        }
    )

    # 音声データのカラムをAudio型にキャスト
    dataset = dataset.cast_column("audio", Audio())

    # train-data と eval-dataに分割
    dataset = dataset.train_test_split(test_size=0.1) # 10%を評価データに

    print(dataset)

    # データセットをhubにエクスポート
    dataset.push_to_hub(export_name, private=True)
    print(f"Exported dataset to {export_name}")
