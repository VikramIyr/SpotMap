import open3d as o3d
import numpy as np
import logging
import os

def visualize_point_cloud(pcd, window_name="Point Cloud Viewer"):
    """
    Visualize a point cloud with the simple Open3D viewer.
    
    Args:
        pcd (open3d.geometry.PointCloud): The point cloud to visualize
        window_name (str): The title for the visualization window
    """
    logging.info(f"[Visualization] Showing {window_name}")
    o3d.visualization.draw_geometries([pcd], window_name=window_name)

def visualize_point_cloud_with_normals(pcd, window_name="Point Cloud with Normals"):
    """
    Visualize a point cloud with normal vectors displayed.
    
    Args:
        pcd (open3d.geometry.PointCloud): The point cloud to visualize (must have normals)
        window_name (str): The title for the visualization window
    """
    if not pcd.has_normals():
        logging.warning("[Visualization] Point cloud does not have normals. Cannot visualize.")
        return
        
    logging.info(f"[Visualization] Showing {window_name} with normals")
    o3d.visualization.draw_geometries([pcd], window_name=window_name, point_show_normal=True)

def visualize_comparison(pcd_before, pcd_after, window_name="Before/After Comparison"):
    """
    Visualize two point clouds side by side for comparison.
    Colors the first point cloud red and the second one green.
    
    Args:
        pcd_before (open3d.geometry.PointCloud): The "before" point cloud
        pcd_after (open3d.geometry.PointCloud): The "after" point cloud
        window_name (str): The title for the visualization window
    """
    # Create colored copies
    pcd_before_colored = o3d.geometry.PointCloud(pcd_before)
    pcd_after_colored = o3d.geometry.PointCloud(pcd_after)
    
    # Color the point clouds
    pcd_before_colored.paint_uniform_color([1, 0, 0])  # Red
    pcd_after_colored.paint_uniform_color([0, 1, 0])  # Green
    
    # Visualize
    logging.info(f"[Visualization] Showing comparison: {window_name}")
    logging.info(f"[Visualization] Red: Before ({len(pcd_before.points)} points)")
    logging.info(f"[Visualization] Green: After ({len(pcd_after.points)} points)")
    o3d.visualization.draw_geometries([pcd_before_colored, pcd_after_colored], window_name=window_name)

def save_point_cloud(pcd, path):
    """
    Save point cloud to file.
    
    Args:
        pcd (open3d.geometry.PointCloud): The point cloud to save
        path (str): The file path to save to
    """
    # Ensure directory exists
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    o3d.io.write_point_cloud(path, pcd)
    logging.info(f"[✓] Saved point cloud to: {path}")

def colorize_by_height(pcd, min_z=None, max_z=None):
    """
    Colorize point cloud by height (z-coordinate) with a rainbow colormap.
    
    Args:
        pcd (open3d.geometry.PointCloud): The point cloud to colorize
        min_z (float, optional): Minimum z value for color mapping. Defaults to min of data.
        max_z (float, optional): Maximum z value for color mapping. Defaults to max of data.
        
    Returns:
        open3d.geometry.PointCloud: Point cloud with colors applied
    """
    points = np.asarray(pcd.points)
    
    if len(points) == 0:
        logging.warning("[Colorize] Point cloud is empty, cannot colorize.")
        return pcd
    
    if min_z is None:
        min_z = np.min(points[:, 2])
    if max_z is None:
        max_z = np.max(points[:, 2])
    
    # Normalize z values to [0, 1]
    z_range = max_z - min_z
    if z_range == 0:  # Handle flat point clouds
        z_range = 1.0
    normalized_z = (points[:, 2] - min_z) / z_range
    
    # Create rainbow colormap
    colors = np.zeros((len(normalized_z), 3))
    for i, t in enumerate(normalized_z):
        # Rainbow color map
        if t < 0.25:  # Blue to Cyan
            r, g, b = 0, 4*t, 1
        elif t < 0.5:  # Cyan to Green
            r, g, b = 0, 1, 1 + 4*(0.25 - t)
        elif t < 0.75:  # Green to Yellow
            r, g, b = 4*(t - 0.5), 1, 0
        else:  # Yellow to Red
            r, g, b = 1, 1 + 4*(0.75 - t), 0
        colors[i] = [r, g, b]
    
    # Apply colors
    colored_pcd = o3d.geometry.PointCloud(pcd)
    colored_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    logging.info(f"[Colorize] Applied height colormap from {min_z:.3f} to {max_z:.3f}")
    return colored_pcd

def colorize_clusters(pcd, eps=0.02, min_points=10):
    """
    Colorize point cloud by cluster membership (DBSCAN clustering).
    
    Args:
        pcd (open3d.geometry.PointCloud): The point cloud to colorize
        eps (float): DBSCAN epsilon parameter (max distance between points in cluster)
        min_points (int): DBSCAN min_points parameter (min cluster size)
        
    Returns:
        open3d.geometry.PointCloud: Point cloud with cluster-based colors applied
    """
    # Perform DBSCAN clustering
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    
    # Get unique labels (excluding noise points labeled as -1)
    valid_labels = labels[labels >= 0]
    if len(valid_labels) == 0:
        logging.warning("[Colorize] No clusters found, cannot colorize.")
        return pcd
    
    # Create a colorized copy
    colored_pcd = o3d.geometry.PointCloud(pcd)
    
    # Generate distinct colors for each label (including noise as black)
    max_label = labels.max() + 1
    colors = np.zeros((len(labels), 3))
    
    # Set noise points (-1) to black
    colors[labels < 0] = [0, 0, 0]
    
    # Generate distinct colors for each cluster using HSV color space
    for i in range(max_label):
        # Create distinct hues, fully saturated
        hue = i / max_label
        # Convert HSV to RGB (simplified conversion)
        h_i = int(hue * 6)
        f = hue * 6 - h_i
        p = 0
        q = 1 - f
        t = f
        
        if h_i == 0:
            r, g, b = 1, t, p
        elif h_i == 1:
            r, g, b = q, 1, p
        elif h_i == 2:
            r, g, b = p, 1, t
        elif h_i == 3:
            r, g, b = p, q, 1
        elif h_i == 4:
            r, g, b = t, p, 1
        else:
            r, g, b = 1, p, q
        
        # Assign this color to all points in the cluster
        colors[labels == i] = [r, g, b]
    
    colored_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    num_clusters = len(np.unique(valid_labels))
    logging.info(f"[Colorize] Colored {num_clusters} clusters")
    return colored_pcd

if __name__ == "__main__":
    # Simple test if run as standalone
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    # Create a sample point cloud
    test_pcd = o3d.geometry.PointCloud()
    points = np.random.rand(1000, 3) * 2 - 1  # Random points in [-1, 1]^3
    test_pcd.points = o3d.utility.Vector3dVector(points)
    
    # Test visualization
    visualize_point_cloud(test_pcd, "Random Point Cloud")
    
    # Test height colorization
    colored_pcd = colorize_by_height(test_pcd)
    visualize_point_cloud(colored_pcd, "Height-Colored Point Cloud")