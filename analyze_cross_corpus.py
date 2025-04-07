#!/usr/bin/env python3

import json
import os
from collections import defaultdict
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_cross_corpus_results():
    """Load all cross-corpus experiment results."""
    results = []

    # Walk through all result directories
    for feature_type in ["MFCC", "LOGMEL", "HUBERT", "FUSION"]:
        for source_dataset in ["EMODB", "RAVDESS"]:
            result_dir = f"results/{feature_type}/{source_dataset}"
            if not os.path.exists(result_dir):
                continue

            # Find all cross-corpus Excel files
            excel_files = glob(f"{result_dir}/cross_corpus_*.xlsx")
            for excel_file in excel_files:
                try:
                    # Extract parameters from filename
                    params = extract_params_from_filename(excel_file)

                    # Read the Excel file
                    df = pd.read_excel(excel_file)

                    # Get the cross-corpus accuracy
                    cross_corpus_acc = df["accuracy"].iloc[-1]  # Last row usually contains final accuracy

                    # Determine target dataset
                    target_dataset = "RAVDESS" if source_dataset == "EMODB" else "EMODB"

                    # Store results
                    results.append(
                        {
                            "feature_type": feature_type,
                            "source_dataset": source_dataset,
                            "target_dataset": target_dataset,
                            "cross_corpus_acc": cross_corpus_acc,
                            "params": params,
                            "file": excel_file,
                        }
                    )
                except Exception as e:
                    print(f"Error processing {excel_file}: {e}")

    return pd.DataFrame(results)


def extract_params_from_filename(filename):
    """Extract parameters from the Excel filename."""
    params = {}
    basename = os.path.basename(filename)

    # Add your parameter extraction logic here
    # This is just an example
    if "lr" in basename:
        params["learning_rate"] = float(basename.split("lr")[1].split("_")[0])
    if "batch" in basename:
        params["batch_size"] = int(basename.split("batch")[1].split("_")[0])
    # Add more parameter extraction as needed

    return params


def analyze_cross_corpus_performance(results_df):
    """Analyze cross-corpus performance across different configurations."""
    # Group by feature type and calculate mean performance
    feature_performance = results_df.groupby("feature_type")["cross_corpus_acc"].mean()

    # Group by source dataset and calculate mean performance
    dataset_performance = results_df.groupby("source_dataset")["cross_corpus_acc"].mean()

    # Find best configuration
    best_config = results_df.loc[results_df["cross_corpus_acc"].idxmax()]

    return {
        "feature_performance": feature_performance.to_dict(),
        "dataset_performance": dataset_performance.to_dict(),
        "best_config": best_config.to_dict(),
    }


def plot_cross_corpus_results(results_df):
    """Plot cross-corpus performance results."""
    plt.figure(figsize=(15, 10))

    # Plot 1: Feature Type Performance
    plt.subplot(2, 2, 1)
    sns.barplot(data=results_df, x="feature_type", y="cross_corpus_acc")
    plt.title("Cross-Corpus Performance by Feature Type")
    plt.xticks(rotation=45)

    # Plot 2: Source Dataset Performance
    plt.subplot(2, 2, 2)
    sns.barplot(data=results_df, x="source_dataset", y="cross_corpus_acc")
    plt.title("Cross-Corpus Performance by Source Dataset")

    # Plot 3: Feature Type vs Source Dataset
    plt.subplot(2, 2, 3)
    pivot_table = results_df.pivot_table(values="cross_corpus_acc", index="feature_type", columns="source_dataset")
    sns.heatmap(pivot_table, annot=True, cmap="YlOrRd")
    plt.title("Cross-Corpus Performance Matrix")

    plt.tight_layout()
    plt.savefig("cross_corpus_analysis.png")
    plt.close()


def main():
    print("Loading cross-corpus results...")
    results_df = load_cross_corpus_results()

    print("\nAnalyzing cross-corpus performance...")
    analysis = analyze_cross_corpus_performance(results_df)

    print("\nFeature Type Performance:")
    for feature, acc in analysis["feature_performance"].items():
        print(f"{feature}: {acc:.4f}")

    print("\nDataset Performance:")
    for dataset, acc in analysis["dataset_performance"].items():
        print(f"{dataset}: {acc:.4f}")

    print("\nBest Configuration:")
    best = analysis["best_config"]
    print(f"Feature Type: {best['feature_type']}")
    print(f"Source Dataset: {best['source_dataset']}")
    print(f"Target Dataset: {best['target_dataset']}")
    print(f"Cross-Corpus Accuracy: {best['cross_corpus_acc']:.4f}")
    print("Parameters:")
    for param, value in best["params"].items():
        print(f"  {param}: {value}")

    print("\nPlotting results...")
    plot_cross_corpus_results(results_df)

    # Save results to JSON
    with open("cross_corpus_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print("\nAnalysis complete! Results saved to cross_corpus_analysis.json and cross_corpus_analysis.png")


if __name__ == "__main__":
    main()
