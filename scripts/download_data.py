"""MediScan Dataset Download & Extraction Helper.

Automates downloading the Kaggle Chest X-Ray (Pneumonia) dataset.
Source: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
"""

import os
import zipfile
import subprocess
import sys


def download_chest_xray(target_dir: str = "data/raw") -> None:
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "chest-xray-pneumonia.zip")
    dataset_name = "paultimothymooney/chest-xray-pneumonia"

    print("=" * 60)
    print(" MediScan: Chest X-Ray Dataset Downloader")
    print("=" * 60)

    # Check if already extracted
    extracted_folder = os.path.join(target_dir, "chest_xray")
    if os.path.exists(extracted_folder):
        print(f"[Notice] Dataset directory '{extracted_folder}' already exists!")
        return

    # Attempt download via Kaggle CLI
    print(f"\n[1/2] Attempting download via Kaggle CLI ({dataset_name})...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_name, "-p", target_dir],
            check=True
        )
        print("[Success] Kaggle download completed.")
    except Exception as e:
        print(f"[Warning] Automatic Kaggle CLI download failed: {e}")
        print("\nManual Instructions:")
        print("1. Go to: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        print("2. Click 'Download' (archive is ~1.2 GB).")
        print(f"3. Place the downloaded ZIP or extracted 'chest_xray' folder into: {os.path.abspath(target_dir)}")
        return

    # Extract zip file
    if os.path.exists(zip_path):
        print(f"\n[2/2] Extracting '{zip_path}' to '{target_dir}'...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        print("[Success] Extraction complete!")


if __name__ == "__main__":
    download_chest_xray()
