"""
Module: image_processing.py

Contains utilities for:
- Counting color frames
- Loading camera intrinsics
- Depth-to-RGB registration (NumPy and PyTorch)
- Color correction for RGB images
- Simple image quality metrics
"""

import os
import cv2
import json
import numpy as np
import re
import torch

from pathlib import Path
from scipy.spatial.transform import Rotation as R

def count_color_frames(path: Path) -> int:
    """Count RGB frames in the format X.jpg, where X is a number."""
    files = os.listdir(path)
    jpg_files = [f for f in files if re.match(r"\d+\.jpg$", f)]
    return len(jpg_files)

def load_camera_info(
    path: Path
) -> tuple:
    """Load camera intrinsics and resolution from a JSON file."""
    with open(path, 'r') as f:
        camera_info = json.load(f)
    intrinsics = camera_info['intrinsics']
    dimensions = camera_info['dimensions']
    fx, fy = intrinsics[0], intrinsics[4]
    cx, cy = intrinsics[2], intrinsics[5]
    width, height = dimensions[0], dimensions[1]

    return fx, fy, cx, cy, width, height

def register_depth_to_rgb(
    depth_image: np.ndarray,
    depth_intrinsics: dict,
    rgb_intrinsics: dict,
    depth_to_rgb: np.ndarray
) -> np.ndarray:
    """
    Register a depth image to the RGB camera frame using the same logic as depth_image_proc/register.cpp.

    Args:
        depth_image (np.ndarray): 2D depth image (uint16, in mm).
        depth_intrinsics (dict): Keys: fx, fy, cx, cy, Tx, Ty.
        rgb_intrinsics (dict): Keys: fx, fy, cx, cy, Tx, Ty.
        depth_to_rgb (np.ndarray): 4x4 affine transformation from depth frame to RGB frame.

    Returns:
        np.ndarray: Registered depth image in the RGB frame (same resolution as RGB).
    """

    # Extract camera parameters
    fx_d, fy_d = depth_intrinsics['fx'], depth_intrinsics['fy']
    cx_d, cy_d = depth_intrinsics['cx'], depth_intrinsics['cy']
    Tx_d, Ty_d = depth_intrinsics.get('Tx', 0.0), depth_intrinsics.get('Ty', 0.0)

    fx_rgb, fy_rgb = rgb_intrinsics['fx'], rgb_intrinsics['fy']
    cx_rgb, cy_rgb = rgb_intrinsics['cx'], rgb_intrinsics['cy']
    Tx_rgb, Ty_rgb = rgb_intrinsics.get('Tx', 0.0), rgb_intrinsics.get('Ty', 0.0)

    inv_fx_d = 1.0 / fx_d
    inv_fy_d = 1.0 / fy_d

    height, width = depth_image.shape
    rgb_height, rgb_width = rgb_intrinsics['height'], rgb_intrinsics['width']
    registered = np.zeros((rgb_height, rgb_width), dtype=np.uint16)

    for v in range(height):
        for u in range(width):
            raw_depth = depth_image[v, u]
            if raw_depth == 0:
                continue

            depth_m = raw_depth / 1000.0  # Convert to meters

            x_d = ((u - cx_d) * depth_m - Tx_d) * inv_fx_d
            y_d = ((v - cy_d) * depth_m - Ty_d) * inv_fy_d
            pt_d = np.array([x_d, y_d, depth_m, 1.0])

            pt_rgb = depth_to_rgb @ pt_d
            z = pt_rgb[2]
            if z <= 0:
                continue

            inv_z = 1.0 / z
            u_rgb = int((fx_rgb * pt_rgb[0] + Tx_rgb) * inv_z + cx_rgb + 0.5)
            v_rgb = int((fy_rgb * pt_rgb[1] + Ty_rgb) * inv_z + cy_rgb + 0.5)

            if 0 <= u_rgb < rgb_width and 0 <= v_rgb < rgb_height:
                new_depth = int(z * 1000.0)  # back to mm
                current = registered[v_rgb, u_rgb]
                if current == 0 or new_depth < current:
                    registered[v_rgb, u_rgb] = new_depth

    return registered

def register_depth_to_rgb_torch(
    depth_image: torch.Tensor,
    depth_intrinsics: dict,
    rgb_intrinsics: dict,
    depth_to_rgb: torch.Tensor,
    min_depth: int = 300,
    max_depth: int = 3000
) -> torch.Tensor:
    """
    Register a depth image to the RGB camera frame using the same logic as depth_image_proc/register.cpp,
    but fully vectorized in PyTorch.

    Args:
        depth_image (torch.Tensor): 2D depth image (H×W), dtype=torch.uint16, in mm.
        depth_intrinsics (dict): Keys: fx, fy, cx, cy, Tx, Ty.
        rgb_intrinsics (dict): Keys: fx, fy, cx, cy, Tx, Ty, height, width.
        depth_to_rgb (torch.Tensor): 4×4 affine transform from depth frame to RGB frame.

    Returns:
        torch.Tensor: Registered depth image in the RGB frame (H_r×W_r), dtype=torch.uint16.
    """
    device = depth_image.device
    # Intrinsics
    fx_d, fy_d = depth_intrinsics['fx'], depth_intrinsics['fy']
    cx_d, cy_d = depth_intrinsics['cx'], depth_intrinsics['cy']
    Tx_d, Ty_d = depth_intrinsics.get('Tx', 0.0), depth_intrinsics.get('Ty', 0.0)

    fx_rgb, fy_rgb = rgb_intrinsics['fx'], rgb_intrinsics['fy']
    cx_rgb, cy_rgb = rgb_intrinsics['cx'], rgb_intrinsics['cy']
    Tx_rgb, Ty_rgb = rgb_intrinsics.get('Tx', 0.0), rgb_intrinsics.get('Ty', 0.0)

    H, W = depth_image.shape
    H_r, W_r = rgb_intrinsics['height'], rgb_intrinsics['width']

    # Convert depth to meters and mask out zeros
    depth_image_int = depth_image.to(torch.int32)
    valid = (depth_image_int >= min_depth) & (depth_image_int <= max_depth)
    depth_m = depth_image.to(torch.float32) / 1000.0

    # Create pixel coordinate grid
    u = torch.arange(0, W, device=device).view(1, -1).expand(H, W)
    v = torch.arange(0, H, device=device).view(-1, 1).expand(H, W)

    # Back-project to depth camera frame
    inv_fx_d = 1.0 / fx_d
    inv_fy_d = 1.0 / fy_d

    x_d = ((u - cx_d) * depth_m - Tx_d) * inv_fx_d
    y_d = ((v - cy_d) * depth_m - Ty_d) * inv_fy_d
    ones = torch.ones_like(depth_m)

    # Stack into homogeneous (4×N)
    pts_d = torch.stack([x_d, y_d, depth_m, ones], dim=0).view(4, -1)

    # Transform into RGB camera frame
    pts_rgb = depth_to_rgb.to(device) @ pts_d  # (4×4) × (4×N) → (4×N)
    x_r, y_r, z_r = pts_rgb[0], pts_rgb[1], pts_rgb[2]

    # Only keep points in front of camera
    valid = valid.view(-1) & (z_r > 0)
    if valid.sum() == 0:
        return torch.zeros((H_r, W_r), dtype=torch.uint16, device=device)

    z_r = z_r[valid]
    x_r = x_r[valid]
    y_r = y_r[valid]

    # Project into RGB image
    inv_z = 1.0 / z_r
    u_r = ((fx_rgb * x_r + Tx_rgb) * inv_z + cx_rgb).round().long()
    v_r = ((fy_rgb * y_r + Ty_rgb) * inv_z + cy_rgb).round().long()

    # Keep only points that land inside RGB frame
    in_bounds = (
        (u_r >= 0) & (u_r < W_r) &
        (v_r >= 0) & (v_r < H_r)
    )
    u_r = u_r[in_bounds]
    v_r = v_r[in_bounds]
    z_r = z_r[in_bounds]

    # Convert back to mm (integers)
    depth_vals = (z_r * 1000.0).round().to(torch.int32)

    # Flattened index into registered image
    idx = v_r * W_r + u_r   # shape = (#valid_pts,)

    # Initialize with a large sentinel so scatter_reduce_ can take the min
    sentinel = torch.full((H_r * W_r,), 65535, dtype=torch.int32, device=device)

    # Scatter-reduce: for each target pixel, keep the minimum depth
    sentinel.scatter_reduce_(
        dim=0,
        index=idx,
        src=depth_vals,
        reduce="amin",
        include_self=True
    )

    # Reshape back, map sentinel→0, and cast to uint16
    registered = sentinel.view(H_r, W_r)
    registered[registered == 65535] = 0
    return registered.to(torch.uint16)
