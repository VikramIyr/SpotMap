#!/bin/bash

# Define the base directory dynamically (assumes the script is run from its own directory)
BASE_DIR="$(pwd)"

# Update paths to use the base directory
python "$BASE_DIR/utils/prepare_o3d_input.py"
python "$BASE_DIR/../../Open3D/examples/python/reconstruction_system/run_system.py" --make --register --refine --slac --slac_integrate --config "$BASE_DIR/../../configs/spot.json"
python "$BASE_DIR/utils/extract_pose_from_slam.py"
python "$BASE_DIR/utils/downsample.py"
python "$BASE_DIR/tsdf_reconstruction.py" --relative_path data/data_o3d_ds --pose_subdir pose --output_path tsdf.ply
python "$BASE_DIR/utils/align_ply_z_up.py"
python "$BASE_DIR/tsdf_reconstruction.py" --relative_path data/data_o3d_ds --pose_subdir pose_z_up --output_path tsdf_z_up.ply
python "$BASE_DIR/ply_processing/pc_denoising.py"