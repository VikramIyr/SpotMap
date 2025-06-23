# Placeholder for tag detection
import os
import sys
import cv2
import utils
import apriltag
import contextlib
import numpy as np
import open3d as o3d

from pathlib import Path
from tqdm import tqdm

APRILTAG_SIZE = 0.146

@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = os.dup(2)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stderr, 2)

def visualize_frame_with_fiducial_and_reference(
    color_img: np.ndarray,
    depth_img: np.ndarray,
    intrinsics: np.ndarray,
    T_fiducial_world: np.ndarray,
    T_camera_world: np.ndarray
):
    # 1. Create RGBD image
    color_o3d = o3d.geometry.Image(color_img)
    depth_o3d = o3d.geometry.Image(depth_img)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1000.0,  # depth in millimeters
        convert_rgb_to_intensity=False
    )

    # 2. Create Open3D intrinsics
    height, width = depth_img.shape
    fx, fy, cx, cy = intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2]
    o3d_intr = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # 3. Generate point cloud and transform to world frame
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intr)
    pcd.transform(T_camera_world)  # now in world frame

    # 4. Create coordinate frames
    frame_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    frame_fiducial = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1).transform(T_fiducial_world)

    # 5. Visualize all together
    o3d.visualization.draw_geometries([pcd, frame_fiducial, frame_world])

def detect_apriltag(
    data_path: str,
    tag_id: int = 52,
    tag_size: float = APRILTAG_SIZE,  # meters
    detector: apriltag.Detector = None,
    visualize: bool = True
):
    image_path = os.path.join(data_path, "color")
    depth_path = os.path.join(data_path, "depth")

    if detector is None:
        options = apriltag.DetectorOptions(
            families='tag36h11', 
            refine_pose=True,
            refine_edges=True,
            refine_decode=True
        )
        detector = apriltag.Detector(options)

    n_frames = utils.count_frames_in_folder(
        folder=image_path,
        frame_pattern=r"frame_\d{5}\.jpg$"
    )

    best_fit, best_frame_idx, best_detection = -1, -1, None

    for frame in range(n_frames):
        image_file = os.path.join(image_path, f"frame_{frame:05d}.jpg")
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        # with suppress_stderr():
        detections = detector.detect(image)

        for detection in detections:
            if detection.tag_id == tag_id and detection.decision_margin > best_fit:
                best_fit = detection.decision_margin
                best_frame_idx = frame
                best_detection = detection

        if frame == 0:
            progress_bar = tqdm(total=n_frames, desc="Searching best AprilTag", unit="frame")
        progress_bar.update(1)
    progress_bar.close()

    if best_detection is None:
        raise NoFiducialDetectedError()

    # Load best frame data
    image_file = os.path.join(image_path, f"frame_{best_frame_idx:05d}.jpg")
    depth_file = os.path.join(depth_path, f"frame_{best_frame_idx:05d}.png")
    json_file  = os.path.join(image_path, f"frame_{best_frame_idx:05d}.json")

    color = cv2.imread(image_file)
    depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
    fx, fy, cx, cy = utils.load_intrinsics_from_json(json_file, title="intrinsics")
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    T_camera_world = utils.load_pose_from_json(json_file, title="cameraPoseARFrame")

    # Prepare 2D–3D correspondence
    s = tag_size / 2.0
    obj_pts = np.array([
        [-s, -s, 0], [ s, -s, 0], [ s,  s, 0], [-s,  s, 0]
    ], dtype=np.float32)
    img_pts = best_detection.corners.astype(np.float32)

    center_2D = best_detection.center.astype(int)
    z = depth[center_2D[1], center_2D[0]] / 1000.0  # mm to meters

    if z == 0:
        tvec_guess = np.zeros((3, 1))
        rvec_guess = None
        use_guess = False
    else:
        x = (center_2D[0] - cx) * z / fx
        y = (center_2D[1] - cy) * z / fy
        tvec_guess = np.array([[x], [y], [z]])
        rvec_guess = cv2.Rodrigues(np.linalg.inv(best_detection.homography))[0]
        use_guess = True

    success, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        intrinsics,
        np.zeros((4, 1)),
        rvec=rvec_guess,
        tvec=tvec_guess,
        useExtrinsicGuess=use_guess,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        raise RuntimeError("PnP failed")

    R_tag, _ = cv2.Rodrigues(rvec)
    T_camera_tag = np.eye(4)
    T_camera_tag[:3, :3] = R_tag
    T_camera_tag[:3, 3] = tvec.squeeze()

    T_world_tag = T_camera_world @ T_camera_tag

    if visualize:
        visualize_frame_with_fiducial_and_reference(
            color_img=color,
            depth_img=depth,
            intrinsics=intrinsics,
            T_fiducial_world=T_world_tag,
            T_camera_world=T_camera_world
        )

    return T_world_tag, best_frame_idx

def detect_apriltag_o3d(
    data_path: str,
    intrinsics: np.ndarray,
    tag_id: int = 52,
    tag_size: float = APRILTAG_SIZE,  # Default tag size in meters
    detector: apriltag.Detector = None,
    visualize: bool = True
):
    if detector is None:
        options = apriltag.DetectorOptions(
            families='tag36h11', 
            refine_pose=True,
            refine_edges=True,
            refine_decode=True
        )
        detector = apriltag.Detector(options)

    image_path = os.path.join(data_path, "image")
    depth_path = os.path.join(data_path, "depth")
    pose_path  = os.path.join(data_path, "poses")

    n_frames = utils.count_frames_in_folder(
        folder=image_path,
        frame_pattern=r"\d+\.jpg$"
    )

    best_fit, best_frame_idx, best_detection = -1, -1, None

    for frame in range(n_frames):
        image_file = os.path.join(image_path, f"{frame}.jpg")
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        detections = detector.detect(image)

        for detection in detections:
            if detection.tag_id == tag_id and detection.decision_margin > best_fit:
                best_fit = detection.decision_margin
                best_frame_idx = frame
                best_detection = detection

        if frame == 0:
            progress_bar = tqdm(total=n_frames, desc="Searching best AprilTag", unit="frame")
        progress_bar.update(1)
    progress_bar.close()

    if best_detection is None:
        raise RuntimeError("No matching AprilTag found.")

    # Load best frame
    image_file = os.path.join(image_path, f"{best_frame_idx}.jpg")
    depth_file = os.path.join(depth_path, f"{best_frame_idx}.png")
    pose_file  = os.path.join(pose_path, f"{best_frame_idx}.txt")

    color = cv2.imread(image_file)
    depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
    T_camera_world = utils.load_pose_from_txt(pose_file)

    # Prepare 2D–3D correspondences
    s = tag_size / 2.0
    obj_pts = np.array([
        [-s, -s, 0], [ s, -s, 0], [ s,  s, 0], [-s,  s, 0]
    ], dtype=np.float32)
    img_pts = best_detection.corners.astype(np.float32)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    center_2D = best_detection.center.astype(int)
    z = depth[center_2D[1], center_2D[0]] / 1000.0  # mm to m

    if z == 0:
        tvec_guess = np.zeros((3, 1))
        rvec_guess = None
        use_guess = False
    else:
        x = (center_2D[0] - cx) * z / fx
        y = (center_2D[1] - cy) * z / fy
        tvec_guess = np.array([[x], [y], [z]])
        rvec_guess = cv2.Rodrigues(np.linalg.inv(best_detection.homography))[0]
        use_guess = True

    success, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        intrinsics,
        np.zeros((4, 1)),
        rvec=rvec_guess,
        tvec=tvec_guess,
        useExtrinsicGuess=use_guess,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        raise RuntimeError("PnP failed")

    R_tag, _ = cv2.Rodrigues(rvec)
    T_camera_tag = np.eye(4)
    T_camera_tag[:3, :3] = R_tag
    T_camera_tag[:3, 3] = tvec.squeeze()

    T_world_tag = T_camera_world @ T_camera_tag

    if visualize:
        visualize_frame_with_fiducial_and_reference(
            color_img=color,
            depth_img=depth,
            intrinsics=intrinsics,
            T_fiducial_world=T_world_tag,
            T_camera_world=T_camera_world
        )

    return T_world_tag, best_frame_idx

class NoFiducialDetectedError(Exception):
    pass