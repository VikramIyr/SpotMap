import os
import shutil

from pathlib import Path


def subsample_dataset(dataset_dir, output_dir, mod_factor):
    """
    Subsamples an RGB-D dataset by copying every N-th frame from the input directories to new output directories.

    Args:
        dataset_dir (str or Path): Root directory of the original dataset.
        output_dir (str or Path): Directory to store the subsampled dataset.
        mod_factor (int): Copy every N-th frame.
    """
    color_dir = Path(dataset_dir) / "image"
    depth_dir = Path(dataset_dir) / "depth"
    pose_dir  = Path(dataset_dir) / "pose_z_up"

    out_color = Path(output_dir) / "image"
    out_depth = Path(output_dir) / "depth"
    out_pose  = Path(output_dir) / "pose"

    for d in [out_color, out_depth, out_pose]:
        d.mkdir(parents=True, exist_ok=True)

    frame_ids = sorted([
        int(f.split(".")[0]) for f in os.listdir(color_dir)
        if f.endswith(".jpg") and f.split(".")[0].isdigit()
    ])

    n_copied = 0
    for idx in frame_ids:
        if idx % mod_factor != 0:
            continue

        name = f"{idx}"
        jpg_path = color_dir / f"{name}.jpg"
        png_path = depth_dir / f"{name}.png"
        txt_path = pose_dir / f"{name}.txt"

        if jpg_path.exists() and png_path.exists() and txt_path.exists():
            sub_idx = idx // mod_factor
            shutil.copy(jpg_path, out_color / f"{sub_idx}.jpg")
            shutil.copy(png_path, out_depth / f"{sub_idx}.png")
            shutil.copy(txt_path, out_pose / f"{sub_idx}.txt")
            n_copied += 1
        else:
            print(f"[WARN] Missing file(s) for frame {name}, skipping.")

    print(f"[INFO] Subsampled {n_copied} frames using modulo {mod_factor}.")

if __name__ == "__main__":
    # Example paths - update these as needed
    root = Path(__file__).parent.parent
    dataset_dir = root / "data" / "data_o3d"
    output_dir = root / "data" / "data_om3"
    mod_factor = 5
    subsample_dataset(dataset_dir, output_dir, mod_factor)