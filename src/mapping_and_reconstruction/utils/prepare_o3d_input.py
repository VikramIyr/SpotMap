"""
Utility to prepare Open3D reconstruction input by copying image and depth data
from 'data_extracted' to 'data_o3d', and generating Open3D-compatible camera_intrinsic.json.
"""

import shutil
import json
from pathlib import Path

def prepare_open3d_input(root_dir: Path):
    src_dir = root_dir / "data" / "data_extracted"
    dst_dir = root_dir / "data" / "data_o3d"

    # Define subfolders
    src_image_dir = src_dir / "image"
    src_depth_dir = src_dir / "depth_processed"
    dst_image_dir = dst_dir / "image"
    dst_depth_dir = dst_dir / "depth"

    # Create target folders
    dst_image_dir.mkdir(parents=True, exist_ok=True)
    dst_depth_dir.mkdir(parents=True, exist_ok=True)

    # Copy JPG images
    for jpg_file in sorted(src_image_dir.glob("*.jpg")):
        shutil.copy(jpg_file, dst_image_dir / jpg_file.name)

    # Copy PNG depth maps
    for png_file in sorted(src_depth_dir.glob("*.png")):
        shutil.copy(png_file, dst_depth_dir / png_file.name)

    # Load camera info from 0.json
    meta_file = src_image_dir / "0.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"Missing camera metadata: {meta_file}")

    with meta_file.open("r") as f:
        meta = json.load(f)

    width, height = meta["dimensions"]
    intrinsics = meta["intrinsics"]

    # Format for Open3D
    intrinsic_o3d = {
        "width": width,
        "height": height,
        "intrinsic_matrix": [
            intrinsics[0], intrinsics[1], intrinsics[3],
            intrinsics[6], intrinsics[4], intrinsics[7],
            intrinsics[2], intrinsics[5], intrinsics[8]
        ]
    }

    with open(dst_dir / "camera_intrinsic.json", "w") as f:
        json.dump(intrinsic_o3d, f, indent=2)

    print(f"Copied {len(list(dst_image_dir.glob('*.jpg')))} images to {dst_image_dir}")
    print(f"Copied {len(list(dst_depth_dir.glob('*.png')))} depth maps to {dst_depth_dir}")
    print(f"Saved Open3D intrinsics to {dst_dir / 'camera_intrinsic.json'}")


if __name__ == "__main__":
    ROOT = Path(__file__).parent.parent.parent.parent
    prepare_open3d_input(ROOT)
