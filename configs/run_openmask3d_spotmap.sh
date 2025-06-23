#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
set -e

# RUN OPENMASK3D FOR A SINGLE SCENE
# This script performs the following:
# 1. Compute class agnostic masks and save them
# 2. Compute mask features for each mask and save them

# --------
# NOTE: SET THESE PARAMETERS BASED ON YOUR SCENE!
# data paths
SCENE_DIR="$(pwd)/resources/data_om3"
SCENE_POSE_DIR="${SCENE_DIR}/pose"
SCENE_INTRINSIC_PATH="${SCENE_DIR}/intrinsic/intrinsic_color.txt"
SCENE_INTRINSIC_RESOLUTION="[240,320]" # change if your intrinsics are based on another resolution
SCENE_PLY_PATH="${SCENE_DIR}/scene.ply"
SCENE_COLOR_IMG_DIR="${SCENE_DIR}/color"
SCENE_DEPTH_IMG_DIR="${SCENE_DIR}/depth"
IMG_EXTENSION=".jpg"
DEPTH_EXTENSION=".png"
DEPTH_SCALE=1000
# model ckpt paths
MASK_MODULE_CKPT_PATH="$(pwd)/resources/scannet200_benchmark.ckpt"
SAM_CKPT_PATH="$(pwd)/resources/sam_vit_h_4b8939.pth"
# output directories to save masks and mask features
MASK_SAVE_DIR="$(pwd)/outputs/data_om3/masks" 
MASK_FEATURE_SAVE_DIR="$(pwd)/outputs/data_om3/mask_features"
SAVE_VISUALIZATIONS=false #if set to true, saves pyviz3d visualizations
# gpu optimization
OPTIMIZE_GPU_USAGE=false

cd openmask3d

# 1. Compute class agnostic masks and save them
echo "[INFO] Extracting class agnostic masks..."
python class_agnostic_mask_computation/get_masks_single_scene.py \
general.experiment_name="single_scene" \
general.checkpoint=${MASK_MODULE_CKPT_PATH} \
general.train_mode=false \
data.test_mode=test \
model.num_queries=120 \
general.use_dbscan=true \
general.dbscan_eps=0.95 \
general.save_visualizations=${SAVE_VISUALIZATIONS} \
general.scene_path=${SCENE_PLY_PATH} \
general.mask_save_dir=${MASK_SAVE_DIR}
echo "[INFO] Mask computation done!"

# get the path of the saved masks
MASK_FILE_BASE=$(echo $SCENE_PLY_PATH | sed 's:.*/::')
MASK_FILE_NAME=${MASK_FILE_BASE/.ply/_MASKS.pt}
SCENE_MASK_PATH="${MASK_SAVE_DIR}/${MASK_FILE_NAME}"
echo "[INFO] Masks saved to ${SCENE_MASK_PATH}."

# 2. Compute mask features for each mask and save them
echo "[INFO] Computing mask features..."

python compute_features_single_scene.py \
data.masks.masks_path=${SCENE_MASK_PATH} \
data.camera.poses_path=${SCENE_POSE_DIR} \
data.camera.intrinsic_path=${SCENE_INTRINSIC_PATH} \
data.camera.intrinsic_resolution=${SCENE_INTRINSIC_RESOLUTION} \
data.depths.depths_path=${SCENE_DEPTH_IMG_DIR} \
data.depths.depth_scale=${DEPTH_SCALE} \
data.images.images_path=${SCENE_COLOR_IMG_DIR} \
data.point_cloud_path=${SCENE_PLY_PATH} \
output.output_directory=${MASK_FEATURE_SAVE_DIR} \
external.sam_checkpoint=${SAM_CKPT_PATH} \
gpu.optimize_gpu_usage=${OPTIMIZE_GPU_USAGE}
#echo "[INFO] Feature computation done!"
