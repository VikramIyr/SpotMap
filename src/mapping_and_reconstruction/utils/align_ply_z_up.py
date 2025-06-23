import open3d as o3d
import numpy as np
from pathlib import Path
from scipy.spatial import ConvexHull

def load_point_cloud(file_path):
    return o3d.io.read_point_cloud(file_path)

def detect_floor_plane(pcd):
    plane_model, inliers = pcd.segment_plane(0.02, 3, 1000)
    return plane_model, pcd.select_by_index(inliers)

def align_normal_to_negative_y(plane_model):
    normal = np.array(plane_model[:3])
    normal /= np.linalg.norm(normal)
    target = np.array([0, -1, 0])
    dot = np.dot(normal, target)

    if np.abs(dot) > 0.999999:
        R = np.eye(3) if dot > 0 else np.diag([-1, -1, 1])
    else:
        axis = np.cross(normal, target)
        axis /= np.linalg.norm(axis)
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K

    T = np.eye(4)
    T[:3, :3] = R
    return T

def align_rectangle_with_axes(pcd):
    _, inliers = detect_floor_plane(pcd)
    pts = np.asarray(inliers.points)[:, [0, 2]]

    hull = ConvexHull(pts)
    best_angle, best_R = None, None
    min_area = np.inf

    for angle in np.linspace(0, np.pi/2, 180):
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
        rotated = pts[hull.vertices] @ R.T
        w, h = np.ptp(rotated, axis=0)
        area = w * h
        if area < min_area:
            min_area = area
            best_angle = angle
            best_R = R

    align_angle = -best_angle if abs(best_angle) < abs(best_angle - np.pi/2) else -(best_angle - np.pi/2)
    Ry = np.array([[np.cos(align_angle), 0, np.sin(align_angle)],
                   [0, 1, 0],
                   [-np.sin(align_angle), 0, np.cos(align_angle)]])
    T = np.eye(4)
    T[:3, :3] = Ry
    return T

def rotate_y_to_z():
    Rx = np.eye(4)
    Rx[1:3, 1:3] = [[0, -1], [1, 0]]
    return Rx

def rotate_180_z():
    Rz = np.eye(4)
    Rz[:2, :2] = [[-1, 0], [0, -1]]
    return Rz

def load_pose(path: Path) -> np.ndarray:
    return np.loadtxt(path).reshape(4, 4)

def save_pose(path: Path, pose: np.ndarray):
    np.savetxt(path, pose, fmt="%.6f")

def main():
    root_dir = Path(__file__).parent.parent.parent.parent
    tsdf_path = root_dir / "data" / "data_o3d_ds" / "tsdf.ply"
    poses_dir = root_dir / "data" / "data_o3d_ds" / "pose"
    out_dir = root_dir / "data" / "data_o3d_ds" / "pose_z_up"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Align scene and compute transformation
    pcd = load_point_cloud(str(tsdf_path))
    plane_model, _ = detect_floor_plane(pcd)
    T1 = align_normal_to_negative_y(plane_model)
    pcd.transform(T1)

    T2 = align_rectangle_with_axes(pcd)
    pcd.transform(T2)

    T3 = rotate_y_to_z()
    pcd.transform(T3)

    T4 = rotate_180_z()
    pcd.transform(T4)

    # Total transformation: T = T4 @ T3 @ T2 @ T1
    T_total = T4 @ T3 @ T2 @ T1

    # Step 2: Apply to poses
    pose_files = sorted(poses_dir.glob("*.txt"))
    for pose_path in pose_files:
        pose = load_pose(pose_path)
        transformed = T_total @ pose
        out_path = out_dir / pose_path.name
        save_pose(out_path, transformed)

    print(f"Transformed {len(pose_files)} poses → {out_dir}")

if __name__ == "__main__":
    main()
