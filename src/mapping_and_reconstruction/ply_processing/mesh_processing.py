import open3d as o3d

# Load your mesh
mesh = o3d.io.read_triangle_mesh("/home/cvg-robotics/project9_ws/Spot-Xplore-Robotic-Scene-Understanding-by-Exploration/project-9/data_open3d_downsampled/scene.ply")

# 1. Clean up
mesh.remove_duplicated_vertices()
mesh.remove_duplicated_triangles()
mesh.remove_unreferenced_vertices()
mesh.remove_degenerate_triangles()
mesh.remove_non_manifold_edges()

# 2. Orient and normalize
mesh.orient_triangles()
mesh.compute_vertex_normals()

# 3. Optional: Smoothing (moderate iterations to preserve structure)
# print('filter with Taubin with 5 iterations')
# mesh = mesh.filter_smooth_taubin(number_of_iterations=5)
# o3d.visualization.draw_geometries([mesh], window_name="After Smoothing")

# Save
o3d.io.write_triangle_mesh("/home/cvg-robotics/project9_ws/Spot-Xplore-Robotic-Scene-Understanding-by-Exploration/project-9/data_open3d/scene_cleaned.ply", mesh)
