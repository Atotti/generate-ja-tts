import argparse
from src.format_to_datasets import format_to_datasets

def main():
    # ArgumentParser を作成
    parser = argparse.ArgumentParser(
        description="A tool to process and format datasets."
    )

    # サブコマンドを作成
    subparsers = parser.add_subparsers(dest="command", required=True)

    # "format" コマンド
    format_parser = subparsers.add_parser(
        "format",
        help="Format data into datasets"
    )
    format_parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input file"
    )
    format_parser.add_argument(
        "--export_name",
        type=str,
        required=True,
        help="Name for the exported dataset"
    )

    # 引数を解析
    args = parser.parse_args()

    # コマンドごとの処理
    if args.command == "format":
        print(f"Formatting datasets from {args.input_file} to {args.export_name}...")
        format_to_datasets(args.input_file, args.export_name)

if __name__ == "__main__":
    main()
