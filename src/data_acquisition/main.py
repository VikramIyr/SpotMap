"""
main.py — Orchestrates data extraction from rosbag and depth preprocessing.

Steps:
1. Extract RGB, depth, pose, and camera info from rosbag
2. Resize and register depth to RGB frame
3. Apply optional filtering
4. Save processed outputs
"""

import os
import torch
import time
import cv2
import numpy as np

from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from image_processing import (
    count_color_frames,
    load_camera_info,
    register_depth_to_rgb,
    register_depth_to_rgb_torch
)
from rosbag_processing import (
    read_and_save_image_topic,
    read_and_save_pose_topic,
    read_save_camera_info
)

def main():
    ROOT_DIR = Path(__file__).parent.parent.parent
    ROSBAG_DIR = ROOT_DIR / 'src' / 'data_acquisition'/ 'ros2' / 'rgbd_dataset'
    DATASET_DIR = ROOT_DIR / 'data' / 'data_extracted'
    COLOR_DIR = DATASET_DIR / 'image'
    DEPTH_DIR = DATASET_DIR / 'depth'
    DEPTH_PROCESSED_DIR = DATASET_DIR / 'depth_processed'

    # Create necessary folders
    for d in [COLOR_DIR, DEPTH_DIR, DEPTH_PROCESSED_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Rosbag extraction
    print("\nExtracting data from rosbag...")
    read_and_save_image_topic(
        topic='/rgb_synced/image',
        rosbag_dir=ROSBAG_DIR,
        save_dir=COLOR_DIR,
        extension='jpg',
        save_timestamp=True
    )
    read_and_save_image_topic(
        topic='/depth_synced/image',
        rosbag_dir=ROSBAG_DIR,
        save_dir=DEPTH_DIR,
        extension='png',
        save_timestamp=True
    )
    read_and_save_pose_topic(
        topic='/camera_pose',
        rosbag_dir=ROSBAG_DIR,
        save_dir=COLOR_DIR
    )
    read_save_camera_info(
        topic='/rgb_synced/camera_info',
        rosbag_dir=ROSBAG_DIR,
        save_dir=COLOR_DIR
    )
    read_save_camera_info(
        topic='/depth_synced/camera_info',
        rosbag_dir=ROSBAG_DIR,
        save_dir=DEPTH_DIR
    )

    print("Processing raw depth images...")
    n_frames = count_color_frames(COLOR_DIR)
    cuda_available = torch.cuda.is_available()

    for i in tqdm(range(n_frames), desc="Registering depth"):
        color_info_path = COLOR_DIR / f"{i}.json"
        depth_info_path = DEPTH_DIR / f"{i}.json"
        depth_path = DEPTH_DIR / f"{i}.png"

        color_fx, color_fy, color_cx, color_cy, color_width, color_height = load_camera_info(color_info_path)
        depth_fx, depth_fy, depth_cx, depth_cy, depth_width, depth_height = load_camera_info(depth_info_path)

        scale_x = color_width / depth_width
        scale_y = color_height / depth_height

        depth_image = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            continue

        resized_depth = cv2.resize(depth_image, (color_width, color_height), interpolation=cv2.INTER_NEAREST)
        depth_fx *= scale_x
        depth_fy *= scale_y
        depth_cx *= scale_x
        depth_cy *= scale_y

        T_depth_to_rgb = np.eye(4)
        T_depth_to_rgb[:3, :3] = R.from_euler('xyz', [0.0, 0.0, 90.0], degrees=True).as_matrix()
        T_depth_to_rgb[:3, 3] = np.array([0.020, 0.016, -0.006])

        if cuda_available:
            depth_tensor = torch.from_numpy(resized_depth).to("cuda")
            T_tensor = torch.from_numpy(T_depth_to_rgb).float().to("cuda")
            registered_depth = register_depth_to_rgb_torch(
                depth_tensor,
                {
                    'fx': depth_fx, 'fy': depth_fy,
                    'cx': depth_cx, 'cy': depth_cy,
                    'Tx': 0.0, 'Ty': 0.0
                },
                {
                    'fx': color_fx, 'fy': color_fy,
                    'cx': color_cx, 'cy': color_cy,
                    'Tx': 0.0, 'Ty': 0.0,
                    'width': color_width, 'height': color_height
                },
                T_tensor
            ).cpu().numpy()
        else:
            registered_depth = register_depth_to_rgb(
                resized_depth,
                {
                    'fx': depth_fx, 'fy': depth_fy,
                    'cx': depth_cx, 'cy': depth_cy,
                    'Tx': 0.0, 'Ty': 0.0
                },
                {
                    'fx': color_fx, 'fy': color_fy,
                    'cx': color_cx, 'cy': color_cy,
                    'Tx': 0.0, 'Ty': 0.0,
                    'width': color_width, 'height': color_height
                },
                T_depth_to_rgb
            )

        filtered = cv2.medianBlur(registered_depth, ksize=3)
        cv2.imwrite(str(DEPTH_PROCESSED_DIR / f"{i}.png"), filtered)

if __name__ == "__main__":
    main()