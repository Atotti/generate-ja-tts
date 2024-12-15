import argparse
from src.format_to_datasets import format_to_datasets
from src.cuda_check import check_cuda
from src.play_model import gen
from src.gen_high_score import gen_high_score
from src.ecapa_confusion_matrix import confusion_matrix

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

    # "cuda-check" コマンド
    cuda_check_parser = subparsers.add_parser(
        "cuda-check",
        help="Check if CUDA is available"
    )

    # "generate" コマンド
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate audio files"
    )
    generate_parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to use"
    )
    generate_parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt for the model"
    )
    generate_parser.add_argument(
        "--description",
        type=str,
        required=True,
        help="Description for the model"
    )
    generate_parser.add_argument(
        "--output_file_path",
        type=str,
        required=True,
        help="Path to the output file"
    )

    # "generate-high-score" コマンド
    generate_high_score_parser = subparsers.add_parser(
        "generate-high-score",
        help="Generate audio files with high score"
    )

    # "plot-ecapa-confusion-matrix"
    plot_ecapa_confusion_matrix_parser = subparsers.add_parser(
        "plot-ecapa-confusion-matrix",
        help="Plot confusion matrix for ecapa model"
    )


    # 引数を解析
    args = parser.parse_args()

    # コマンドごとの処理
    if args.command == "format":
        print(f"Formatting datasets from {args.input_file} to {args.export_name}...")
        format_to_datasets(args.input_file, args.export_name)
    elif args.command == "cuda-check":
        check_cuda()
    elif args.command == "generate":
        print(f"Generating audio files with model {args.model_name}...")
        print(f"Prompt: {args.prompt}\nDescription: {args.description}")
        gen(args.model_name, args.prompt, args.description, args.output_file_path)
        print(f"Audio file generated at {args.output_file_path}")
    elif args.command == "generate-high-score":
        gen_high_score()
    elif args.command == "plot-ecapa-confusion-matrix":
        confusion_matrix()


if __name__ == "__main__":
    main()
