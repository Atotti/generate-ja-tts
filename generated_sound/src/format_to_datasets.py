from datasets import DatasetDict, Audio, Dataset
import os

def format_to_datasets(input_file: str, export_name: str, name: str):
    """Format data into datasets

    Args:
        input_file (str): csv file path
        export_name (str): name of the dataset to export to HuggingFace Hub
        name (str): name of the dataset
    """
    # CSVファイルからデータセットを作成
    dataset = Dataset.from_csv(input_file)

    # csvファイル中の<name>をnameに置換
    dataset = dataset.map(
        lambda x: {
            "audio": x["audio"].replace("<name>", name),
            "text": x["text"]
        }
    )

    # input_file からの相対パスを絶対パスに変換
    dataset = dataset.map(
        lambda x: {
            "audio": os.path.join(os.path.dirname(input_file), x["audio"]),
            "text": x["text"]
        }
    )

    # audioに対応するファイルが無い場合はrowを削除
    dataset = dataset.filter(
        lambda x: os.path.exists(x["audio"]),
    )

    # 音声データのカラムをAudio型にキャスト
    dataset = dataset.cast_column("audio", Audio())

    # train-data と eval-dataに分割
    dataset_split = dataset.train_test_split(test_size=0.1)  # 10%を評価データに

    # DatasetDictに分割データを格納
    dataset_dict = DatasetDict({
        "train": dataset_split["train"],
        "eval": dataset_split["test"]
    })

    print(dataset_dict)

    # データセットをhubにエクスポート
    dataset_dict.push_to_hub(export_name, private=True)
    print(f"Exported dataset to {export_name}")
