from datasets import load_dataset
import argparse

def display_dataset_content(dataset_name, configuration=None):
    """
    データセットをロードし、サンプルを表示します。
    """
    print(f"Loading dataset: {dataset_name}")

    # データセットのロード
    if configuration:
        dataset = load_dataset(dataset_name, configuration)
    else:
        dataset = load_dataset(dataset_name)

    print("\nDataset splits:")
    for split in dataset.keys():
        print(f" - {split}: {len(dataset[split])} samples")

    print("\nSample content:")
    for split in dataset.keys():
        print(f"\n=== Split: {split} ===")
        for i, sample in enumerate(dataset[split]):
            print(sample)
            if i >= 2:  # 各スプリットで3サンプルだけ表示
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", type=str, help="Hugging Face HubのリポジトリID (e.g., 'username/repository')")
    parser.add_argument("--configuration", default=None, type=str, help="データセットの設定 (必要であれば指定)")
    parser.add_argument("--save_dir", default=None, type=str, help="データセットをローカルに保存するディレクトリ")

    args = parser.parse_args()

    # データセットを取得して内容を表示
    dataset = load_dataset(args.repo_id, args.configuration)
    display_dataset_content(args.repo_id, args.configuration)

    # ローカルに保存する場合
    if args.save_dir:
        print(f"\nSaving dataset to disk: {args.save_dir}")
        dataset.save_to_disk(args.save_dir)
        print("Dataset saved successfully.")
