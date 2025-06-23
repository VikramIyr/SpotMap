import open3d as o3d
import numpy as np
import logging
import multiprocessing
import os

def keep_top_k_clusters(pcd, eps, min_points, k, voxel_size=None):
    """
    Fast top-k clustering via downsampling + mapping back to original.

    Args:
        pcd (o3d.geometry.PointCloud): the full-resolution input
        eps (float): DBSCAN distance threshold
        min_points (int): DBSCAN min points
        k (int): how many largest clusters to keep
        voxel_size (float, optional): size for voxel_down_sample_and_trace
                                      defaults to eps/2 if not provided
    Returns:
        o3d.geometry.PointCloud: only the points from the top-k clusters
    """
    # choose default voxel_size if missing
    if voxel_size is None:
        voxel_size = eps * 0.5

    os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())

    down_pcd, _, voxel_map = pcd.voxel_down_sample_and_trace(
        voxel_size,
        pcd.get_min_bound(),
        pcd.get_max_bound(),
    )
    logging.info(f"[Cluster] Downsampled from {len(pcd.points)} → {len(down_pcd.points)} points")

    labels = np.array(down_pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
        print_progress=False
    ))
    mask = labels >= 0
    if not np.any(mask):
        raise RuntimeError("No clusters found on downsampled cloud. Try larger eps or smaller voxel_size.")

    counts = np.bincount(labels[mask])
    top_k_labels = np.argsort(counts)[-k:]
    logging.info(f"[Cluster] Top-{k} labels (downsample): {top_k_labels.tolist()}")

    selected_voxels = np.where(np.isin(labels, top_k_labels))[0]
    orig_indices = []
    for vid in selected_voxels:
        orig_indices.extend(voxel_map[vid])
    orig_indices = list(set(orig_indices))

    logging.info(f"[Cluster] Selected {len(orig_indices)} original points from top-{k} clusters")

    return pcd.select_by_index(orig_indices)
