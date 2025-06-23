"""
Utility to prepare Open3D reconstruction input by copying image and depth data
from 'data_extracted' to 'data_o3d'.
"""

import shutil
from pathlib import Path

def prepare_open3d_input(root_dir: Path):
    src_dir = root_dir / "data" / "data_extracted"
    dst_dir = root_dir / "data" / "data_o3d"

    # Define subfolders
    src_image_dir = src_dir / "color"
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

    print(f"Copied {len(list(dst_image_dir.glob('*.jpg')))} images to {dst_image_dir}")
    print(f"Copied {len(list(dst_depth_dir.glob('*.png')))} depth maps to {dst_depth_dir}")


if __name__ == "__main__":
    ROOT = Path(__file__).parent.parent
    prepare_open3d_input(ROOT)
