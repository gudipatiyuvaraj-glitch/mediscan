"""MediScan Exploratory Data Analysis (EDA) Script.

Scans the raw Chest X-Ray dataset, computes class distributions,
analyzes resolutions, and saves EDA charts to reports/figures/.
"""

import os
from glob import glob
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image


def run_eda(data_dir: str = "data/raw/chest_xray", output_dir: str = "reports/figures") -> None:
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print(" MediScan: Exploratory Data Analysis (EDA)")
    print("=" * 60)

    # In the Kaggle dataset, the structure is chest_xray/{train, val, test}/{NORMAL, PNEUMONIA}
    splits = ["train", "val", "test"]
    stats = {}

    total_images = 0
    all_classes = ["NORMAL", "PNEUMONIA"]

    for split in splits:
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            print(f"[Warning] Split folder '{split_path}' not found.")
            continue

        stats[split] = {}
        for cls_name in all_classes:
            pattern = os.path.join(split_path, cls_name, "*.*")
            files = [f for f in glob(pattern) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            count = len(files)
            stats[split][cls_name] = count
            total_images += count

    if not stats:
        print("[Error] No dataset found. Please download the dataset into data/raw/chest_xray first.")
        return

    print("\nDataset Class Distribution:")
    print("-" * 50)
    for split, counts in stats.items():
        print(f"Split: {split.upper():<6} | NORMAL: {counts.get('NORMAL', 0):<5} | PNEUMONIA: {counts.get('PNEUMONIA', 0):<5} | Total: {sum(counts.values())}")
    print(f"\nTotal Dataset Images: {total_images}")

    # Plot Class Distribution Bar Chart
    plt.figure(figsize=(8, 5))
    splits_found = list(stats.keys())
    normal_counts = [stats[s].get("NORMAL", 0) for s in splits_found]
    pneumonia_counts = [stats[s].get("PNEUMONIA", 0) for s in splits_found]

    x = range(len(splits_found))
    width = 0.35

    plt.bar([i - width/2 for i in x], normal_counts, width=width, label="NORMAL", color="#2ecc71")
    plt.bar([i + width/2 for i in x], pneumonia_counts, width=width, label="PNEUMONIA", color="#e74c3c")

    plt.xlabel("Dataset Split")
    plt.ylabel("Number of Images")
    plt.title("Chest X-Ray Class Distribution by Split")
    plt.xticks(x, [s.upper() for s in splits_found])
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    dist_chart_path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(dist_chart_path, dpi=300)
    plt.close()
    print(f"[Saved] Class distribution chart saved to '{dist_chart_path}'")


if __name__ == "__main__":
    run_eda()
