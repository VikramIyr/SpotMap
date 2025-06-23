## OpenMask3D Setup and Usage Guide

Refer to the [Spot-Compose GitHub repository](https://github.com/oliver-lemke/spot-compose) for details on Docker container setup.

Activate the `open3d_gpu` Conda environment:
```bash
conda activate open3d_gpu
```

Prepare the data by running the following script:
```bash
python utils/prepare_om3_input.py
```

### Step 1: Start the Container
Run the following command to start the OpenMask3D Docker container:
```bash
docker run -p 5001:5001 --gpus all --name openmask3d_container -it craiden/openmask:v1.0
```

### Step 2: Copy Data and Configuration to the Container
Transfer the required data and configuration files to the container:
```bash
docker cp /home/cvg-robotics/project9_ws/SpotMap/data/data_om3 openmask3d_container:/home/openmask/resources

docker cp /home/cvg-robotics/project9_ws/SpotMap/configs/run_openmask3d_spotmap.sh openmask3d_container:/home/openmask/
```

### Step 3: Run the Segmentation Script Inside the Container
Execute the segmentation script within the container:
```bash
cd openmask
bash run_openmask3d_spotmap.sh
```

### Step 4: Copy Masks and Features to the Local Host
Retrieve the generated masks and features from the container to the local host:
```bash
docker cp openmask3d_container:/home/openmask/outputs/data_om3/mask_features/2025-06-23/experiment_0/clip_features.npy /home/cvg-robotics/project9_ws/SpotMap/data/data_om3/openmask3d/

docker cp openmask3d_container:/home/openmask/outputs/data_om3/masks/scene_MASKS.pt /home/cvg-robotics/project9_ws/SpotMap/data/data_om3/openmask3d/
```

## Scene Graph Setup and Interaction

### Step 1: Activate the Conda Environment
Activate the previously installed `open3d_gpu` Conda environment:
```bash
conda activate open3d_gpu
```

### Step 2: Run the Scene Graph Script
Ensure the script is executable and run it:
```bash
chmod +x run.sh
bash run.sh
```

### Step 3: Interaction Module: Demo
Run the interaction module with the following example command:
```bash
python scene_graph/update_scene_graph.py --pointcloud /home/cvg-robotics/project9_ws/SpotMap/scene_graph/scene.ply --labels_csv /home/cvg-robotics/project9_ws/SpotMap/scene_graph/color_label_mapping.csv --actions translate left 20 "White plastic bottle"
```