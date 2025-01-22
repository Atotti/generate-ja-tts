import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datasets import load_dataset

def analyze_audio_duration(dataset_name=None, data_num=None, target=None, output_dir="eval_dataset_results"):
    """
    Analyze total audio duration in a dataset and visualize the results.

    Args:
        dataset_name (str): Hugging Face dataset name.
        data_num (int): Number of samples to process.
        output_dir (str): Directory to save results.
    """
    os.makedirs(output_dir, exist_ok=True)

    if dataset_name:
        # Load dataset from Hugging Face
        dataset = load_dataset(dataset_name, split="train")
        audio_samples = dataset if data_num is None else dataset[:data_num]  # Get specified number of samples

        # Calculate durations manually
        durations = []
        for sample in audio_samples["audio"]:
            sampling_rate = sample["sampling_rate"]
            array_length = len(sample["array"])
            duration = array_length / sampling_rate  # Calculate duration in seconds
            durations.append(duration)
    else:
        raise ValueError("A dataset name must be provided.")

    # Convert durations to a pandas DataFrame
    duration_df = pd.DataFrame({"Sample": range(len(durations)), "Duration (s)": durations})

    # Compute statistics
    statistics = {
        "Total Duration (s)": np.sum(durations),
    }

    # Save statistics to a text file
    stats_file = os.path.join(output_dir, "audio_duration.txt" if target is None else f"audio_duration_{target}.txt")
    with open(stats_file, "w") as f:
        f.write("Audio Total Duration\n")
        f.write("==========================\n")
        for stat_name, value in statistics.items():
            f.write(f"{stat_name}: {value:.2f}\n")

    print(f"Analysis complete. Results saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Hugging Face dataset name")
    parser.add_argument("--data_num", type=int, help="Number of samples to analyze")
    parser.add_argument("--target", type=str, help="Target label for classification")
    args = parser.parse_args()

    analyze_audio_duration(args.dataset, args.data_num, args.target)
