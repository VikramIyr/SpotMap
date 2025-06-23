import open3d as o3d
import numpy as np
import os
import re
import json
import logging
import argparse

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def count_color_frames(color_folder):
    files = os.listdir(color_folder)
    jpg_files = [f for f in files if re.match(r"\d+\.jpg$", f)]
    return len(jpg_files)

def load_pose(pose_file):
    with open(pose_file, "r") as f:
        lines = f.readlines()
    pose_values = [float(x) for line in lines for x in line.strip().split()]
    T = np.array(pose_values).reshape(4, 4)
    T = np.linalg.inv(T)
    return T

def load_intrinsics_from_pose_json(pose_path):
    with open(pose_path, "r") as f:
        pose_json = json.load(f)
    intrinsics = pose_json["intrinsics"]
    fx, fy = intrinsics[0], intrinsics[4]
    cx, cy = intrinsics[2], intrinsics[5]
    width, height = int(2 * cx), int(2 * cy)
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
    return intrinsic

def integrate_frame(volume, color_path, depth_path, pose_path, intrinsic, depth_scale, depth_trunc, greyscale_output):
    color_raw = o3d.io.read_image(color_path)
    depth_raw = o3d.io.read_image(depth_path)
    camera_T_world = load_pose(pose_path)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=greyscale_output
    )

    volume.integrate(rgbd, intrinsic, camera_T_world)

def extract_and_save_point_cloud(volume, output_path):
    logging.info("Extracting final point cloud...")
    pcd = volume.extract_point_cloud()
    o3d.io.write_point_cloud(output_path, pcd)
    logging.info(f"Point cloud saved to: {output_path}")
    return pcd

def tsdf_reconstruction(relative_path: str, pose_subdir: str, output_path: str):
    base_root = "/home/cvg-robotics/project9_ws/SpotMap"
    data_dir = os.path.join(base_root, relative_path)

    color_subdir = "image"
    depth_subdir = "depth"

    color_dir = os.path.join(data_dir, color_subdir)
    n_frames = count_color_frames(color_dir)
    logging.info(f"Detected {n_frames} frames in the color folder.")

    depth_scale = 1000.0
    depth_trunc = 3.0
    voxel_size = 0.005
    sdf_trunc = 0.1
    greyscale_output = False

    if not o3d.core.cuda.is_available():
        logging.warning("CUDA is NOT available in Open3D. Using CPU.")
    else:
        logging.info("CUDA is available.")

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(640, 480, 552.0291012161067, 552.0291012161067, 320.0, 240.0)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    for i in range(n_frames):
        color_path = os.path.join(data_dir, color_subdir, f"{i}.jpg")
        depth_path = os.path.join(data_dir, depth_subdir, f"{i}.png")
        pose_path = os.path.join(data_dir, pose_subdir, f"{i}.txt")

        if not all(map(os.path.exists, [color_path, depth_path, pose_path])):
            logging.warning(f"Skipping missing frame {i}")
            continue

        integrate_frame(volume, color_path, depth_path, pose_path, intrinsic,
                        depth_scale, depth_trunc, greyscale_output)
        logging.info(f"Integrated frame {i}")

    pcd = extract_and_save_point_cloud(volume, os.path.join(data_dir, output_path))
    o3d.visualization.draw_geometries([pcd], window_name="TSDF Reconstructed Point Cloud")

def main():
    parser = argparse.ArgumentParser(description="TSDF Reconstruction with Open3D")
    parser.add_argument("--relative_path", type=str, required=True, help="Relative path from base root to the scene directory")
    parser.add_argument("--pose_subdir", type=str, default="pose_z_up", help="Subdirectory containing the pose files")
    parser.add_argument("--output_path", type=str, default="tsdf_z_up.ply", help="Output filename for the final point cloud")

    args = parser.parse_args()
    tsdf_reconstruction(args.relative_path, args.pose_subdir, args.output_path)

if __name__ == "__main__":
    main()
