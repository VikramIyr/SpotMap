import open3d as o3d

# Load your mesh
mesh = o3d.io.read_triangle_mesh("/home/cvg-robotics/project9_ws/Spot-Xplore-Robotic-Scene-Understanding-by-Exploration/project-9/data_open3d/scene_3_mesh.ply")

# Visualize the original mesh
o3d.visualization.draw_geometries([mesh], window_name="Original Mesh")