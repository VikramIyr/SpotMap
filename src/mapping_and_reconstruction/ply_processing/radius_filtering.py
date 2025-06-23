import open3d as o3d
import logging

def remove_radius_outliers(pcd, nb_points, radius):
    """
    Remove outliers based on radius search.
    
    This function removes points that have fewer than nb_points neighbors
    within a sphere of the specified radius. It's effective for removing
    isolated points while preserving the structure of the point cloud.
    
    Args:
        pcd (open3d.geometry.PointCloud): The input point cloud
        nb_points (int): Minimum number of points within radius to keep a point
        radius (float): Radius of the sphere to search for neighbors
    
    Returns:
        open3d.geometry.PointCloud: Point cloud with outliers removed
    """
    pcd_filtered, ind = pcd.remove_radius_outlier(
        nb_points=nb_points, 
        radius=radius
    )
    logging.info(f"[ROR] Retained {len(ind)} points after radius filtering.")
    return pcd_filtered

if __name__ == "__main__":
    # Simple test if run as standalone
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    # Create a sample point cloud with outliers
    test_pcd = o3d.geometry.PointCloud()
    
    # Create a dense cluster of points
    cluster_points = [[i*0.1, j*0.1, k*0.1] for i in range(10) for j in range(10) for k in range(10)]
    
    # Add some isolated outliers
    outliers = [[5, 5, 5], [10, 10, 10], [-5, -5, -5], [7, -7, 3]]
    
    all_points = cluster_points + outliers
    test_pcd.points = o3d.utility.Vector3dVector(all_points)
    
    # Apply radius outlier removal
    logging.info(f"Original point cloud has {len(test_pcd.points)} points")
    filtered = remove_radius_outliers(test_pcd, nb_points=5, radius=0.2)
    logging.info(f"Filtered point cloud has {len(filtered.points)} points")
    
    # Visualize the result
    o3d.visualization.draw_geometries([filtered], window_name="Radius Filtering Test")