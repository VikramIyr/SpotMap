#!/bin/bash

python openmask3d/inference.py
python scene_graph/generate_scene_graph.py --pointcloud /home/cvg-robotics/project9_ws/SpotMap/scene_graph/scene.ply --labels_csv /home/cvg-robotics/project9_ws/SpotMap/scene_graph/color_label_mapping.csv --visualize --labels 