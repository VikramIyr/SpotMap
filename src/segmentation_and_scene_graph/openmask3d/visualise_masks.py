import numpy as np
import open3d as o3d
import torch
import clip
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from constants import *
import argparse

def assign_color_to_single_mask(mask, background_color=(0.77, 0.77, 0.77), mask_color=(1.0, 0.0, 0.0)):
    num_points = mask.shape[0]
    colors = np.ones((num_points, 3)) * background_color
    colors[mask > 0.5, :] = mask_color
    return colors

class MaskVisualizer:
    def __init__(self, pcd, masks):
        self.scene_pcd = pcd
        self.masks = masks
        self.num_masks = masks.shape[0]
        self.current_mask_idx = 0
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.pcd_vis = o3d.geometry.PointCloud(pcd)  # copy

    def update_mask(self, idx):
        mask = self.masks[idx]
        color = plt.colormaps.get_cmap("nipy_spectral")(idx / self.num_masks)[:3]
        mask_colors = assign_color_to_single_mask(mask, mask_color=color)
        self.pcd_vis.colors = o3d.utility.Vector3dVector(mask_colors)
        self.vis.update_geometry(self.pcd_vis)
        print(f"Showing mask {idx}")

    def run(self):
        self.vis.create_window()
        self.vis.add_geometry(self.pcd_vis)
        self.update_mask(self.current_mask_idx)

        def next_mask(vis):
            self.current_mask_idx = (self.current_mask_idx + 1) % self.num_masks
            self.update_mask(self.current_mask_idx)
            return False

        self.vis.register_key_callback(ord("A"), next_mask)
        self.vis.run()
        self.vis.destroy_window()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-split", choices=["q1", "q2", "q3", "q4", "all"], default="all")
    args = parser.parse_args()

    ROOT_DIR = "/home/cvg-robotics/project9_ws/SpotMap/data/data_om3"
    DATA_DIR = f"{ROOT_DIR}"
    OM3D_DIR = f"{ROOT_DIR}/openmask3d"

    scene_path = f"{DATA_DIR}/scene.ply"
    mask_path = f"{OM3D_DIR}/scene_MASKS.pt"

    scene_pcd = o3d.io.read_point_cloud(scene_path)
    pred_masks = np.asarray(torch.load(mask_path, weights_only=False)).T  # shape: (num_masks, num_points)

    num_masks = pred_masks.shape[0]
    quarter = num_masks // 4

    if args.mask_split == "q1":
        pred_masks = pred_masks[:quarter]
    elif args.mask_split == "q2":
        pred_masks = pred_masks[quarter:2*quarter]
    elif args.mask_split == "q3":
        pred_masks = pred_masks[2*quarter:3*quarter]
    elif args.mask_split == "q4":
        pred_masks = pred_masks[3*quarter:]

    vis = MaskVisualizer(scene_pcd, pred_masks)
    vis.run()

if __name__ == "__main__":
    main()
