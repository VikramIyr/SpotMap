"""
Utility to prepare OpenMask3D reconstruction input 
from 'data_o3d_ds' to 'data_om3'.
"""

import shutil
from pathlib import Path

def prepare_open3d_input(root_dir: Path):
    src_dir = root_dir / "data" / "data_o3d_ds"
    dst_dir = root_dir / "data" / "data_om3"

    # Define subfolders
    src_image_dir = src_dir / "image"
    src_depth_dir = src_dir / "depth"
    src_pose_dir = src_dir / "pose_z_up"
    src_ply_path = src_dir / "scene_denoised.ply"
    dst_image_dir = dst_dir / "color"
    dst_depth_dir = dst_dir / "depth"
    dst_pose_dir = dst_dir / "pose"
    dst_intrinsic_dir = dst_dir / "intrinsic"
    dst_ply_path = dst_dir / "scene.ply"
    dst_openmask3d_dir = dst_dir / "openmask3d"

    # Create target folders
    dst_image_dir.mkdir(parents=True, exist_ok=True)
    dst_depth_dir.mkdir(parents=True, exist_ok=True)
    dst_pose_dir.mkdir(parents=True, exist_ok=True)
    dst_intrinsic_dir.mkdir(parents=True, exist_ok=True)
    dst_openmask3d_dir.mkdir(parents=True, exist_ok=True)

    # Copy JPG images
    for jpg_file in sorted(src_image_dir.glob("*.jpg")):
        shutil.copy(jpg_file, dst_image_dir / jpg_file.name)

    # Copy PNG depth maps
    for png_file in sorted(src_depth_dir.glob("*.png")):
        shutil.copy(png_file, dst_depth_dir / png_file.name)

    # Copy pose files
    for pose_file in sorted(src_pose_dir.glob("*.txt")):
        shutil.copy(pose_file, dst_pose_dir / pose_file.name)

    # Copy the denoised point cloud
    if src_ply_path.exists():
        shutil.copy(src_ply_path, dst_ply_path)
    
    # Write intrinsic parameters to a file
    intrinsic_file = dst_intrinsic_dir / "intrinsic_color.txt"

    with open(intrinsic_file, "w") as f:
        f.write("552.0291012161067 0.0 320.0 0.0\n")
        f.write("0.0 552.0291012161067 240.0 0.0\n")
        f.write("0.0 0.0 1.0 0.0\n")
        f.write("0.0 0.0 0.0 1.0\n")    

    print(f"Copied {len(list(dst_image_dir.glob('*.jpg')))} images to {dst_image_dir}")
    print(f"Copied {len(list(dst_depth_dir.glob('*.png')))} depth maps to {dst_depth_dir}")
    print(f"Copied {len(list(dst_pose_dir.glob('*.txt')))} pose files to {dst_pose_dir}")
    print(f"Copied point cloud to {dst_ply_path}")


if __name__ == "__main__":
    ROOT = Path(__file__).parent.parent.parent.parent
    prepare_open3d_input(ROOT)
