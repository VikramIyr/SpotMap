import open3d as o3d
import numpy as np

# Load original mesh
# mesh = o3d.io.read_triangle_mesh("/home/cvg-robotics/project9_ws/Spot-Xplore-Robotic-Scene-Understanding-by-Exploration/project-9/data_open3d/scene_cleaned.ply")
# extent = mesh.get_axis_aligned_bounding_box().get_extent()
# max_extent = max(extent)

# # Estimate good Poisson depth
# voxel_desired = 0.01  # Desired resolution: 1cm
# depth = int(np.log2(max_extent / voxel_desired))
# print(f"Suggested Poisson depth: {depth}")

# Prepare point cloud (already filtered and clustered)
# Load the point cloud
pcd = o3d.io.read_point_cloud("/home/cvg-robotics/project9_ws/Spot-Xplore-Robotic-Scene-Understanding-by-Exploration/project-9/data_open3d_downsampled_rotated/scene.ply")

print()
print("Estimating normals...")
pcd.estimate_normals()

# Poisson reconstruction
print()
print("Starting Poisson reconstruction...")
mesh_new, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
print("Finished Poisson reconstruction with depth 10")

# Density-based trimming
densities = np.asarray(densities)
threshold = np.quantile(densities, 0.005)
mesh_new.remove_vertices_by_mask(densities < threshold)

# Final cleanup
mesh_new.remove_duplicated_vertices()
mesh_new.remove_degenerate_triangles()
mesh_new.remove_unreferenced_vertices()
mesh_new.orient_triangles()
mesh_new.compute_vertex_normals()

# Visualize the result
o3d.visualization.draw_geometries([mesh_new], window_name="Poisson Reconstruction Result")

# Save the cleaned mesh
o3d.io.write_triangle_mesh("/home/cvg-robotics/project9_ws/Spot-Xplore-Robotic-Scene-Understanding-by-Exploration/project-9/data_open3d_downsampled_rotated/mesh.ply", mesh_new)