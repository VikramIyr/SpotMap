import os
import icp
import fiducial
import numpy as np
import open3d as o3d

from pathlib import Path


def register_point_clouds_via_apriltag(
    root_dir    : Path,
    data_dir_1  : Path,
    data_dir_2  : Path,
    output_dir  : Path = None,
    debug       : bool = False
):
    """
    Register two point clouds using AprilTag detection.

    Parameters:
        root_dir    : Root directory of the project.
        data_dir_1  : Directory of the first dataset.
        data_dir_2  : Directory of the second dataset.
        output_dir  : Directory to save the combined point cloud, default is the first dataset directory.
        debug       : If True, enables debug mode with additional visualizations.
    """

    # Detect best AprilTag frames in both datasets
    print("Starting AprilTag detection...")
    print()

    if debug:

        print("Dataset Directory 1:", data_dir_1.relative_to(root_dir))
        T_world_tag_1, best_frame_idx_1 = fiducial.detect_apriltag(
            data_path=str(data_dir_1)
        )

        print()
        print("Dataset Directory 2:", data_dir_2.relative_to(root_dir))
        T_world_tag_2, best_frame_idx_2 = fiducial.detect_apriltag(
            data_path=str(data_dir_2)
        )

    else:
        print("Dataset Directory 1:", data_dir_1.relative_to(root_dir))
        T_world_tag_1, best_frame_idx_1 = fiducial.detect_apriltag(
            data_path=str(data_dir_1),
            visualize=False
        )

        print()
        print("Dataset Directory 2:", data_dir_2.relative_to(root_dir))
        T_world_tag_2, best_frame_idx_2 = fiducial.detect_apriltag(
            data_path=str(data_dir_2),
            visualize=False
        )

    # Load point clouds
    scene_1_path = data_dir_1 / "scene.ply"
    scene_2_path = data_dir_2 / "scene.ply"

    pcd_1 = o3d.io.read_point_cloud(str(scene_1_path))
    pcd_2 = o3d.io.read_point_cloud(str(scene_2_path))

    # Compute initial alignment using the AprilTag poses
    T_align = T_world_tag_1 @ np.linalg.inv(T_world_tag_2)

    if debug:
        icp.evaluate_registration(
            source=pcd_2,
            target=pcd_1,
            transformation=T_align,
            threshold=0.05,  # meters
            print_results=True
        )

        icp.draw_registration_result(
            source=pcd_2,
            target=pcd_1,
            transformation=T_align
        )

    # Register the right point cloud to the left using ICP
    pcd_2.transform(T_align)

    # Apply ICP refinement
    print("Starting ICP refinement...")
    T, _, _ = icp.apply_point_to_plane_icp(
        source=pcd_2,
        target=pcd_1,
        threshold=0.05,  # meters
    )

    pcd_combined = pcd_1 + pcd_2

    # Save the combined point cloud
    if output_dir is None:
        output_dir = data_dir_1

    combined_pcd_path = output_dir / "combined_scene.ply"
    o3d.io.write_point_cloud(str(combined_pcd_path), pcd_combined)

    return T_align, best_frame_idx_1, best_frame_idx_2

def register_point_clouds_via_apriltag_o3d(
    root_dir    : Path,
    target_path : Path,
    source_path : Path,
    intrinsics  : np.ndarray,
    output_dir  : Path = None,
    trans_poses : bool = False,
    debug       : bool = False
):
    """
    Register two point clouds using AprilTag detection.

    Parameters:
        root_dir    : Root directory of the project.
        data_dir_1  : Directory of the first dataset.
        data_dir_2  : Directory of the second dataset.
        intrinsics  : Camera intrinsics matrix.
        output_dir  : Directory to save the combined point cloud, default is the first dataset directory.
        debug       : If True, enables debug mode with additional visualizations.
    """

    # Detect best AprilTag frames in both datasets
    print("Starting AprilTag detection...")
    print()

    if debug:

        print("Dataset Directory 1:", target_path.relative_to(root_dir))
        T_world_tag_1, best_frame_idx_1 = fiducial.detect_apriltag_o3d(
            data_path=str(target_path),
            intrinsics=intrinsics
        )

        print()
        print("Dataset Directory 2:", source_path.relative_to(root_dir))
        T_world_tag_2, best_frame_idx_2 = fiducial.detect_apriltag_o3d(
            data_path=str(source_path),
            intrinsics=intrinsics
        )

    else:
        print("Dataset Directory 1:", target_path.relative_to(root_dir))
        T_world_tag_1, best_frame_idx_1 = fiducial.detect_apriltag_o3d(
            data_path=str(target_path),
            intrinsics=intrinsics,
            visualize=False
        )

        print()
        print("Dataset Directory 2:", source_path.relative_to(root_dir))
        T_world_tag_2, best_frame_idx_2 = fiducial.detect_apriltag_o3d(
            data_path=str(source_path),
            intrinsics=intrinsics,
            visualize=False
        )

    # Load point clouds
    scene_1_path = target_path / "scene.ply"
    scene_2_path = source_path / "scene.ply"

    pcd_1 = o3d.io.read_point_cloud(str(scene_1_path))
    pcd_2 = o3d.io.read_point_cloud(str(scene_2_path))

    # Compute initial alignment using the AprilTag poses
    T_align = T_world_tag_1 @ np.linalg.inv(T_world_tag_2)

    if debug:
        icp.evaluate_registration(
            source=pcd_2,
            target=pcd_1,
            transformation=T_align,
            threshold=0.05,  # meters
            print_results=True
        )

        icp.draw_registration_result(
            source=pcd_2,
            target=pcd_1,
            transformation=T_align
        )

    # Apply ICP refinement
    print("Starting ICP refinement...")
    T_icp,_, _ = icp.apply_point_to_plane_icp(
        source=pcd_2,
        target=pcd_1,
        threshold=0.05,
        trans_init=T_align,
    )

    T_final = T_icp

    pcd_2.transform(T_final)

    pcd_combined = pcd_1 + pcd_2
    o3d.visualization.draw_geometries([pcd_combined], window_name="Combined Point Cloud")

    if trans_poses:
        poses_dir_2 = source_path / "poses"

        transform_pose_files(
            input_dir=poses_dir_2,
            output_dir=source_path / "poses_transformed",
            T_final=T_final
        )

    # Save the combined point cloud
    if output_dir is None:
        output_dir = data_dir_1

    combined_pcd_path = output_dir / "combined_scene.ply"
    o3d.io.write_point_cloud(str(combined_pcd_path), pcd_combined)

    return T_align, best_frame_idx_1, best_frame_idx_2

def transform_pose_files(
    input_dir: Path,
    output_dir: Path,
    T_final: np.ndarray
):
    """
    Apply a transformation to all 4x4 pose matrices stored in .txt files in input_dir.
    Save the transformed poses in output_dir with the same filenames.

    Parameters:
        input_dir : Path to folder containing the original pose files (named X.txt).
        output_dir : Path to folder where the transformed pose files will be saved.
        T_final : 4x4 numpy array transformation matrix to apply to each pose.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith(".txt"):
            continue

        input_path = input_dir / filename
        output_path = output_dir / filename

        # Load 4x4 pose matrix
        pose = np.loadtxt(input_path)

        if pose.shape != (4, 4):
            print(f"Skipping {filename}: invalid shape {pose.shape}")
            continue

        # Apply the transformation
        transformed_pose = T_final @ pose

        # Save the transformed matrix
        np.savetxt(output_path, transformed_pose, fmt="%.17f")

    print(f"Transformed poses saved to: {output_dir}")