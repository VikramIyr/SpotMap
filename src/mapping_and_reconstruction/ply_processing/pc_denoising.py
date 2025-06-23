import open3d as o3d
import numpy as np
import os
import logging
import sys
import time
import multiprocessing

# Import modules for each denoising technique
from statistical_filtering import remove_statistical_outliers
from radius_filtering import remove_radius_outliers
from cluster_filtering import keep_top_k_clusters
from normal_estimation import estimate_normals
from visualization_utils import visualize_point_cloud, save_point_cloud

# === Logging and Performance Setup ===
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
n_threads = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(n_threads)

def visualize_removed_points(before_pcd, after_pcd, title="Removed Points"):
    before_pts = np.asarray(before_pcd.points)
    after_pts = np.asarray(after_pcd.points)

    after_set = set(map(tuple, after_pts.round(decimals=6)))
    removed_pts = np.array([pt for pt in before_pts if tuple(np.round(pt, 6)) not in after_set])

    removed = o3d.geometry.PointCloud()
    removed.points = o3d.utility.Vector3dVector(removed_pts)
    removed.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.0, 0.0], (len(removed_pts), 1)))

    print(f"[INFO] Visualizing: {len(removed_pts)} points removed.")
    o3d.visualization.draw_geometries([after_pcd, removed], window_name=title)


def run_denoising(relative_path: str, visualize_steps=True, skip_radius=True):
    total_start_time = time.time()

    base_root = "/home/cvg-robotics/project9_ws/SpotMap/data/data_o3d_ds"
    input_path = os.path.join(base_root, relative_path, "tsdf_z_up.ply")
    output_dir = os.path.join(base_root, relative_path)
    final_output_path = os.path.join(output_dir, "scene_denoised.ply")
    steps_dir = os.path.join(output_dir, "denoising_steps")
    if visualize_steps and not os.path.exists(steps_dir):
        os.makedirs(steps_dir)

    # # Parameters
    # stat_nb_neighbors = 20
    # stat_std_ratio = 2.0
    # radius_nb_points = 61
    # radius = 0.04
    # eps = 0.04
    # min_points = 5
    # top_k_clusters = 2

    # Stricter Parameters
    stat_nb_neighbors = 20       # Increase neighborhood size → more robust stats
    stat_std_ratio = 2.0         # Stricter threshold → remove more outliers

    stat_nb_neighbors_2 = 10      # Small neighborhood → less aggressive
    stat_std_ratio_2 = 4.0        # High tolerance for variation → keep almost everything  

    radius_nb_points = 6        # Require more neighbors → more selective
    radius = 0.03               # Smaller radius → tighter locality

    eps = 0.03                 # Smaller cluster radius → fewer merges
    min_points = 10              # Require larger clusters to keep
    top_k_clusters = 4           # Keep only the largest cluster


    # === Load ===
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"[!] File not found: {input_path}")

    load_start = time.time()
    logging.info(f"[INFO] Loading point cloud from: {input_path}")
    pcd = o3d.io.read_point_cloud(input_path)
    logging.info(f"[INFO] Loaded point cloud with {len(pcd.points)} points in {time.time() - load_start:.2f}s")

    if visualize_steps:
        save_point_cloud(pcd, os.path.join(steps_dir, "1_original.ply"))
        visualize_point_cloud(pcd, "Original Point Cloud")

    # Step 1: Statistical Filtering
    before = pcd
    step_start = time.time()
    pcd = remove_statistical_outliers(before, stat_nb_neighbors, stat_std_ratio)
    logging.info(f"[TIMING] Statistical outlier removal took {time.time() - step_start:.2f}s")

    if visualize_steps:
        save_point_cloud(pcd, os.path.join(steps_dir, "2_statistical_filtered.ply"))
        visualize_removed_points(before, pcd, "Removed by Statistical Outlier Removal")

    # Step 2: Radius Filtering (optional)
    if not skip_radius:
        before = pcd
        step_start = time.time()
        pcd = remove_radius_outliers(before, radius_nb_points, radius)
        logging.info(f"[TIMING] Radius outlier removal took {time.time() - step_start:.2f}s")

        if visualize_steps:
            save_point_cloud(pcd, os.path.join(steps_dir, "3_radius_filtered.ply"))
            visualize_removed_points(before, pcd, "Removed by Radius Outlier Removal")

    # Step 3: Cluster Filtering
    before = pcd
    step_start = time.time()
    pcd = keep_top_k_clusters(before, eps, min_points, top_k_clusters)
    logging.info(f"[TIMING] Cluster filtering took {time.time() - step_start:.2f}s")

    if visualize_steps:
        visualize_removed_points(before, pcd, "Removed by Cluster Filtering")

    
    # Step 4: Normal Estimation (optional)
    # step_start = time.time()
    # pcd = estimate_normals(pcd, knn=30)
    # logging.info(f"[TIMING] Normal estimation took {time.time() - step_start:.2f}s")
        # Step 4: Second Statistical Filtering (post-clustering)
    before = pcd
    step_start = time.time()
    pcd = remove_statistical_outliers(before, stat_nb_neighbors_2, stat_std_ratio_2)
    logging.info(f"[TIMING] Second statistical outlier removal took {time.time() - step_start:.2f}s")


    if visualize_steps:
        save_point_cloud(pcd, os.path.join(steps_dir, "4_statistical_after_clustering.ply"))
        visualize_removed_points(before, pcd, "Removed by 2nd Statistical Outlier Removal")

    # Final Save + Visual
    save_point_cloud(pcd, final_output_path)
    logging.info(f"[✓] Complete denoising pipeline finished in {time.time() - total_start_time:.2f}s")
    visualize_point_cloud(pcd, "Final Denoised Point Cloud")

    return pcd

if __name__ == "__main__":
    if len(sys.argv) > 1:
        rel_path = sys.argv[1]
    else:
        rel_path = input("Enter relative path to point cloud directory: ")

    if len(sys.argv) > 2:
        visualize = sys.argv[2].lower() not in ('false', 'no', '0')
    else:
        visualize = True

    if len(sys.argv) > 3:
        use_radius = sys.argv[3].lower() in ('true', 'yes', '1')
    else:
        use_radius = False

    run_denoising(rel_path, visualize_steps=visualize, skip_radius=not use_radius)
