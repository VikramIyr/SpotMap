import os
import numpy as np

def extract_poses_and_save(log_path, output_dir):
    # Ensure output folder exists
    os.makedirs(output_dir, exist_ok=True)

    with open(log_path, 'r') as f:
        lines = f.readlines()

    assert len(lines) % 5 == 0, "Unexpected format: Each pose should span 5 lines"

    for i in range(0, len(lines), 5):
        header = lines[i].strip().split()
        frame_idx = int(header[0])  # this is the actual frame index

        # Parse the 4x4 matrix
        pose = np.array([
            list(map(float, lines[i + 1].strip().split())),
            list(map(float, lines[i + 2].strip().split())),
            list(map(float, lines[i + 3].strip().split())),
            list(map(float, lines[i + 4].strip().split()))
        ])

        # Save to X.txt
        output_path = os.path.join(output_dir, f"{frame_idx}.txt")
        np.savetxt(output_path, pose, fmt="%.17g")

    print(f"Saved {len(lines)//5} poses to '{output_dir}'.")

# Example usage:
log_file = "/home/cvg-robotics/project9_ws/SpotMap/data/data_o3d/slac/0.025/optimized_trajectory_slac.log"
output_folder = "/home/cvg-robotics/project9_ws/SpotMap/data/data_o3d/pose"
extract_poses_and_save(log_file, output_folder)
