import open3d as o3d
import numpy as np
import logging

def estimate_normals(pcd, knn=30, orientation_reference=None):
    """
    Estimate normals for a point cloud.
    
    This function computes normal vectors for each point in the cloud using
    a k-nearest neighbors approach. Normals can be oriented consistently by
    providing a reference point (typically camera position for consistency).
    
    Args:
        pcd (open3d.geometry.PointCloud): The input point cloud
        knn (int): Number of nearest neighbors to use for normal estimation
        orientation_reference (ndarray, optional): 3D point to orient normals towards
    
    Returns:
        open3d.geometry.PointCloud: Point cloud with normals computed
    """
    logging.info(f"[Normals] Estimating normals with KNN={knn}")
    
    # Estimate normals using KNN search
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    
    # Orient normals towards a reference point if provided
    if orientation_reference is not None:
        pcd.orient_normals_towards_camera_location(orientation_reference)
        logging.info(f"[Normals] Oriented normals towards reference point")
    
    return pcd

if __name__ == "__main__":
    # Simple test if run as standalone
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    # Create a sample point cloud (sphere)
    test_pcd = o3d.geometry.PointCloud()
    
    # Create points on a sphere
    theta = np.linspace(0, 2*np.pi, 30)
    phi = np.linspace(0, np.pi, 20)
    
    points = []
    for t in theta:
        for p in phi:
            x = np.sin(p) * np.cos(t)
            y = np.sin(p) * np.sin(t)
            z = np.cos(p)
            points.append([x, y, z])
    
    test_pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estimate normals
    logging.info(f"Original point cloud has {len(test_pcd.points)} points")
    with_normals = estimate_normals(test_pcd, knn=20)
    
    # Visualize the point cloud with normals
    logging.info("Visualizing point cloud with normals (small blue lines)")
    o3d.visualization.draw_geometries([with_normals], window_name="Normal Estimation Test", 
                                     point_show_normal=True)