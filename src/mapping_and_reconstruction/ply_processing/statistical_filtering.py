import open3d as o3d
import numpy as np
import logging
from tqdm import tqdm

def remove_statistical_outliers(pcd, nb_neighbors, std_ratio):
    """
    Remove outliers using statistical analysis of distances, with a tqdm progress bar.

    This manually computes, for each point, the average distance to its
    nb_neighbors nearest neighbors, then filters out points whose mean
    distance exceeds (global_mean + std_ratio * global_std).

    Args:
        pcd (open3d.geometry.PointCloud): The input point cloud
        nb_neighbors (int): Number of neighbors to consider for each point
        std_ratio (float): Standard deviation ratio threshold

    Returns:
        open3d.geometry.PointCloud: Point cloud with outliers removed
    """
    points = np.asarray(pcd.points)
    n_points = len(points)
    tree = o3d.geometry.KDTreeFlann(pcd)

    # 1) Compute mean distance to neighbors for each point, with a progress bar
    mean_dists = np.zeros(n_points, dtype=float)
    for i in tqdm(range(n_points), desc="[SOR] Computing distances", unit="pt"):
        # search for nb_neighbors+1 because the point itself is included
        _, idx, dists_sq = tree.search_knn_vector_3d(pcd.points[i], nb_neighbors + 1)
        if len(dists_sq) > 1:
            # skip the zero‐distance to itself
            mean_dists[i] = np.mean(np.sqrt(dists_sq[1:]))
        else:
            mean_dists[i] = np.inf

    # 2) Global statistics
    global_mean = mean_dists.mean()
    global_std = mean_dists.std()
    threshold = global_mean + std_ratio * global_std
    logging.info(f"[SOR] Mean = {global_mean:.4f}, Std = {global_std:.4f}, Threshold = {threshold:.4f}")

    # 3) Filter
    keep_mask = mean_dists <= threshold
    ind = np.nonzero(keep_mask)[0]
    pcd_filtered = pcd.select_by_index(ind.tolist())

    logging.info(f"[SOR] Retained {len(ind)} / {n_points} points after statistical filtering.")
    return pcd_filtered

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Create a sample point cloud with outliers
    test_pcd = o3d.geometry.PointCloud()
    pts = [[i, j, k] for i in range(5) for j in range(5) for k in range(5)]
    pts += [[100, 100, 100], [-100, 50, 25], [75, -75, 30]]
    test_pcd.points = o3d.utility.Vector3dVector(pts)

    logging.info(f"Original point cloud has {len(test_pcd.points)} points")
    filtered = remove_statistical_outliers(test_pcd, nb_neighbors=10, std_ratio=2.0)
    logging.info(f"Filtered point cloud has {len(filtered.points)} points")

    o3d.visualization.draw_geometries([filtered], window_name="Statistical Filtering Test")
