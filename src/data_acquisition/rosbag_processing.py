### File: rosbag_processing.py

"""
Module for reading and processing RGB, depth, pose, and camera info topics
from a ROS 2 bag file. Outputs individual image frames and metadata to disk.
"""

import os
import cv2
import json
import numpy as np

from pathlib import Path
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


def update_json_field(
    filepath: Path, 
    field: str, value
) -> None:
    """Update or create a JSON file with the given field and value."""
    if filepath.exists():
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}

    data[field] = value

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def encoding_to_dtype(
    encoding: str
) -> np.dtype:
    """Map ROS encoding strings to NumPy data types."""
    mapping = {
        "bgr8": np.uint8,
        "16UC1": np.uint16,
    }
    if encoding not in mapping:
        raise ValueError(f"Unsupported encoding: {encoding}")
    return mapping[encoding]


def read_and_save_image_topic(
    topic: str, 
    rosbag_dir: Path, 
    image_dir: Path,
    max_iter: int = 10000, 
    dtype: np.dtype = None,
    extension: str = 'png', 
    save_timestamp: bool = False
) -> None:
    """Extract and save image messages from a rosbag topic."""
    with AnyReader([rosbag_dir]) as reader:
        topic_conn = next(c for c in reader.connections if c.topic == topic)
        all_msgs = sorted(reader.messages(connections=[topic_conn]), key=lambda x: x[1])

        for idx, (conn, timestamp, rawdata) in enumerate(tqdm(all_msgs, desc=f"Processing {topic}")):
            if idx >= max_iter:
                break

            msg = reader.deserialize(rawdata, conn.msgtype)
            secs = timestamp // 1_000_000_000
            nsecs = timestamp % 1_000_000_000

            encoding = encoding_to_dtype(msg.encoding) if hasattr(msg, 'encoding') else dtype
            if encoding is None:
                raise RuntimeError("Missing encoding specification.")

            img = np.frombuffer(msg.data, dtype=encoding).reshape(msg.height, msg.width, -1)
            cv2.imwrite(str(save_dir / f'{idx}.{extension}'), img)

            if save_timestamp:
                update_json_field(
                    filepath=save_dir / f'{idx}.json',
                    field='timestamp',
                    value=f"{secs}.{nsecs}"
                )


def read_save_camera_info(
    topic: str, 
    rosbag_dir: Path, 
    save_dir: Path,
    max_iter: int = 10000
) -> None:
    """Extract and save camera intrinsics from a rosbag topic."""
    with AnyReader([rosbag_dir]) as reader:
        topic_conn = next(c for c in reader.connections if c.topic == topic)
        all_msgs = sorted(reader.messages(connections=[topic_conn]), key=lambda x: x[1])

        for idx, (conn, timestamp, rawdata) in enumerate(tqdm(all_msgs, desc=f"Processing {topic} intrinsics")):
            if idx >= max_iter:
                break

            msg = reader.deserialize(rawdata, conn.msgtype)
            update_json_field(save_dir / f'{idx}.json', 'intrinsics', msg.k.tolist())
            update_json_field(save_dir / f'{idx}.json', 'dimensions', [msg.width, msg.height])


def transform_to_matrix(
    msg
) -> np.ndarray:
    """Convert a TransformStamped message to a 4x4 transformation matrix."""
    translation = [msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z]
    rotation = [msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w]
    matrix = np.eye(4)
    matrix[:3, :3] = R.from_quat(rotation).as_matrix()
    matrix[:3, 3] = translation
    return matrix


def read_and_save_pose_topic(
    topic: str, 
    rosbag_dir: Path, 
    save_dir: Path,
    max_iter: int = 10000
) -> None:
    """Extract and save camera poses from a rosbag topic."""
    with AnyReader([rosbag_dir]) as reader:
        topic_conn = next(c for c in reader.connections if c.topic == topic)
        all_msgs = sorted(reader.messages(connections=[topic_conn]), key=lambda x: x[1])

        for idx, (conn, timestamp, rawdata) in enumerate(tqdm(all_msgs, desc=f"Processing {topic} poses")):
            if idx >= max_iter:
                break

            msg = reader.deserialize(rawdata, conn.msgtype)
            pose_matrix = transform_to_matrix(msg).flatten().tolist()
            update_json_field(save_dir / f'{idx}.json', 'cameraPoseARFrame', pose_matrix)
            