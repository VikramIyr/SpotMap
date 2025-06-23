import os
import open3d as o3d
import numpy as np

from pathlib import Path


def align_pointcloud_z_up_and_transform_poses(input_ply_path: Path,
                                              pose_dir: Path,
                                              output_pose_dir: Path,
                                              matrix_out_path: Path):
    """
    Aligns the input point cloud so the dominant plane normal aligns with -Z axis.
    Transforms all camera poses accordingly and saves the transformation matrix.

    Args:
        input_ply_path (Path): Path to the input .ply file.
        pose_dir (Path): Folder containing 4x4 pose text files.
        output_pose_dir (Path): Folder to write transformed pose files.
        matrix_out_path (Path): Path to save the 4x4 transformation matrix as .txt.
    """
    def is_valid_transform(T):
        if not np.allclose(T[3], [0, 0, 0, 1]):
            return False
        R = T[:3, :3]
        return (np.allclose(R @ R.T, np.eye(3), atol=1e-4) and
                np.isclose(abs(np.linalg.det(R)), 1.0, atol=1e-4))

    pcd = o3d.io.read_point_cloud(str(input_ply_path))
    print(f"Loaded point cloud with {len(pcd.points)} points")

    # Step 1: Detect floor plane
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=2000)
    normal = np.array(plane_model[:3])
    normal /= np.linalg.norm(normal)

    # Step 2: Compute rotation matrix
    target = np.array([0, 0, -1])
    cross = np.cross(normal, target)
    dot = np.dot(normal, target)
    if np.linalg.norm(cross) < 1e-6:
        R_align = np.eye(3) if dot > 0 else -np.eye(3)
    else:
        skew = np.array([[0, -cross[2], cross[1]],
                         [cross[2], 0, -cross[0]],
                         [-cross[1], cross[0], 0]])
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        R_align = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)

    T_align = np.eye(4)
    T_align[:3, :3] = R_align

    if not is_valid_transform(T_align):
        raise ValueError("Generated alignment transformation is invalid.")

    np.savetxt(matrix_out_path, T_align, fmt="%.6f")
    print(f"Saved alignment transformation matrix to: {matrix_out_path}")

    # Step 3: Apply to all poses
    output_pose_dir.mkdir(parents=True, exist_ok=True)
    for file in sorted(pose_dir.glob("*.txt")):
        if file.stem.isdigit():
            pose = np.loadtxt(file).reshape(4, 4)
            pose_transformed = T_align @ pose
            out_path = output_pose_dir / file.name
            np.savetxt(out_path, pose_transformed, fmt="%.12f")
    print(f"Transformed poses saved to: {output_pose_dir}")


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    input_ply = root / "data" / "data_o3d" / "scene.ply"
    pose_input = root / "data" / "data_o3d" / "pose"
    pose_output = root / "data" / "data_o3d" / "pose_z_up"
    matrix_out = root / "data" / "reconstruction" / "alignment_matrix.txt"

    align_pointcloud_z_up_and_transform_poses(input_ply, pose_input, pose_output, matrix_out)