import os
import re
import json
import numpy as np
import open3d as o3d

def count_frames_in_folder(
    folder          : str,
    frame_pattern   : str = r"frame_\d{5}\.jpg$"
):
    files = os.listdir(folder)
    matched_files = [f for f in files if re.match(frame_pattern, f)]
    return len(matched_files)

def load_intrinsics_from_json(
    path    : str,
    title   : str = "intrinsics"
):
    with open(path, "r") as f:
        pose_json = json.load(f)
    intrinsics = pose_json["intrinsics"]
    fx, fy = intrinsics[0], intrinsics[4]
    cx, cy = intrinsics[2], intrinsics[5]
    return fx, fy, cx, cy

def load_pose_from_txt(
    pose_file: str
):
    """Loads a 4x4 matrix from a space-separated text file."""
    with open(pose_file, "r") as f:
        lines = f.readlines()
    pose_values = [float(x) for line in lines for x in line.strip().split()]
    T = np.array(pose_values).reshape(4, 4)
    return T

def load_pose_from_json(
    path    : str,
    title   : str = "cameraPoseARFrame"
):
    with open(path, "r") as f:
        pose_json = json.load(f)
    pose_flat = pose_json["cameraPoseARFrame"]
    T_world_camera = np.array(pose_flat).reshape(4, 4)
    return T_world_camera

def create_point_cloud_from_rgbd(
    color_img       : np.ndarray,
    depth_img       : np.ndarray,
    intrinsics      : np.ndarray,
    T_camera_world  : np.ndarray
):
    # 1. Create RGBD image
    color_o3d = o3d.geometry.Image(color_img)
    depth_o3d = o3d.geometry.Image(depth_img)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, 
        depth_o3d,
        depth_scale=1000.0,  # depth in millimeters
        convert_rgb_to_intensity=False
    )

    # 2. Create Open3D intrinsics
    height, width = depth_img.shape
    fx, fy, cx, cy = intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2]
    o3d_intr = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intr)
    pcd.transform(T_camera_world)  # now in world frame

    # 3. Generate point cloud
    return pcd