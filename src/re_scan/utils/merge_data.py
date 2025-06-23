import os
import shutil
from pathlib import Path

def get_sorted_numeric_stems(folder, ext):
    return sorted([
        int(f.stem) for f in folder.glob(f"*.{ext}")
        if f.stem.isdigit()
    ])

def merge_rgbd_datasets(dataset_a, dataset_b):
    dataset_a = Path(dataset_a)
    dataset_b = Path(dataset_b)

    # Define folders
    folders = ['image', 'depth', 'poses']
    a_paths = {f: dataset_a / f for f in folders}
    b_paths = {f: dataset_b / f for f in folders}

    # Get max index in dataset_a
    a_stems = get_sorted_numeric_stems(a_paths['image'], "jpg")
    if not a_stems:
        print("Dataset A is empty.")
        start_index = 0
    else:
        start_index = max(a_stems) + 1

    print(f"Starting index for dataset_b: {start_index}")

    # Get all stems in dataset_b (only numeric)
    b_stems = get_sorted_numeric_stems(b_paths['image'], "jpg")

    copied = 0
    skipped = []

    for stem in b_stems:
        jpg_path  = b_paths['image'] / f"{stem}.jpg"
        png_path  = b_paths['depth'] / f"{stem}.png"
        txt_path  = b_paths['poses'] / f"{stem}.txt"

        if not (jpg_path.exists() and png_path.exists() and txt_path.exists()):
            skipped.append(stem)
            continue

        new_stem = start_index + copied

        shutil.copy(jpg_path,  a_paths['image'] / f"{new_stem}.jpg")
        shutil.copy(png_path,  a_paths['depth'] / f"{new_stem}.png")
        shutil.copy(txt_path,  a_paths['poses'] / f"{new_stem}.txt")
        copied += 1

    print(f"\nCopied {copied} frames from dataset_b to dataset_a.")
    if skipped:
        print(f"Skipped {len(skipped)} incomplete frames: {skipped}")

# Example usage
# merge_rgbd_datasets("/home/cvg-robotics/project9_ws/Spot-Xplore-Robotic-Scene-Understanding-by-Exploration/project-9/data_merged", 
#                     "/home/cvg-robotics/project9_ws/Spot-Xplore-Robotic-Scene-Understanding-by-Exploration/project-9/data_refine_downsampled")
