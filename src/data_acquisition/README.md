# Spot ROS 2
Follow the instructions on the [Spot ROS 2 GitHub repository](https://github.com/bdaiinstitute/spot_ros2) to install Spot ROS 2 Docker.

### Step 1: Start the Container
Run the following command on the host system once per session:
```bash
xhost +local:root
```

Then, start the Docker container:
```bash
docker run -it \
    --runtime=nvidia \
    --gpus all \
    --net=host \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/spot_ros2_logs:/ros_ws/logs \
    spot_ros2:latest
```

Install additional dependencies inside the container:
```bash
sudo apt install ros-humble-rqt*
apt-get install -y nano iputils-ping
```

### Step 2: Start the Spot Driver
Copy the `spot_image_sync` package (located in the `ros2` folder) to the container's `src` directory. This package includes a custom approximate time synchronizer module optimized for Boston Dynamics Spot.

Source the required environments:
```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
clear
```

Launch the Spot driver:
```bash
ros2 launch spot_driver spot_driver.launch.py config_file:=config.yaml launch_rviz:=True launch_image_publishers:=True
```

### Step 3: Install rqt_bag
Install the `rqt_bag` tool:
```bash
sudo apt install ros-humble-rqt-bag
```

### Step 4: Launch the Approximate Time Synchronizer
Run the `rgbd_sync_node` with the following parameters:
```bash
ros2 run spot_image_sync rgbd_sync_node --ros-args \
    -p queue_size:=5 \
    -p max_interval_sec:=0.015 \
    -p age_penalty:=1 \
    -p use_sim_time:=true
```

Alternatively, use these parameters:
```bash
ros2 run spot_image_sync rgbd_sync_node --ros-args \
    -p queue_size:=20 \
    -p max_interval:=0.03 \
    -p age_penalty:=1
```

### Step 5: Record a rosbag
Record the desired topics into a rosbag:
```bash
ros2 bag record /rgb_synced/image /rgb_synced/camera_info /depth_synced/image /depth_synced/camera_info /camera_pose -o rgbd_dataset --start-paused
```

### Step 6: Copy the rosbag to the Host Machine
Transfer the recorded rosbag to the `ros2` folder of this module.

### Extracting Data from rosbag and Processing Raw Depth Maps
Create a Conda environment:
```bash
conda env create -f environment.yml
conda activate cenv-rosbag
```

Run the processing script:
```bash
python main.py
```