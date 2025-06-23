<h1 align="center">
  <img src="docs/images/spotmap_logo.png" alt="SpotMap Logo" width="300" /><br>
  SpotMap
</h1>

<p align="center">
  <strong>Kerem Kılıç<sup>1</sup>, Vikram Iyer<sup>1</sup>, Roham Zendehdel Nobari<sup>1</sup>, Yagiz Devre<sup>2</sup></strong><br>
  <em><small><sup>1</sup>ETH Zürich,&nbsp;,&nbsp;<sup>2</sup>Princeton</small></em>
</p>

---

**SpotMap** is a complete end-to-end pipeline for robotic scene understanding and interactive mapping. It integrates on-board SLAM, volumetric 3D reconstruction, instance segmentation, dynamic scene graph generation, and affordance prediction — giving mobile robots the capability to build detailed maps and understand their surroundings with minimal human supervision.

---


## 📂 Repository Structure

```plaintext
SpotMap/
├── src/          # Core source code: modules of the pipeline
├── configs/      # Configuration files for SLAM and OpenMask3D
├── data/         # Original and intermediate datasets
├── scene_graph/  # Current scene graph representation
├── docs/         # Documentation, figures, and additional resources
```
### 📂 Source code
```plaintext
src/
├── data_acquisition/                
├── mapping_and_reconstruction/
├── segmentation_and_scene_graph/   
└── re_scan/    
```
---

## 🚀 Setup

Each of the core modules in `src/` has its own README with all the installation, configuration, and usage instructions:

- **Data Acquisition**  
  See [`src/data_acquisition/README.md`](src/data_acquisition/README.md) for how to extract and preprocess raw RGB-D frames and ROS bag files.

- **Mapping & Reconstruction**  
  See [`src/mapping_and_reconstruction/README.md`](src/mapping_and_reconstruction/README.md) for how to configure and run SLAM, TSDF fusion, denoising, and mesh/point-cloud export.

- **Segmentation & Scene Graph**  
  See [`src/segmentation_and_scene_graph/README.md`](src/segmentation_and_scene_graph/README.md) for how to perform semantic segmentation, build the scene graph, and visualize the results.

  ### 📂 Dataset

  A sample dataset captured by Spot is already available at [`src/data_acquisition/ros2/rgbd_dataset`](src/data_acquisition/ros2/rgbd_dataset). This dataset can be used to test and validate the pipeline components.


## 📄 Citation

If you find this work useful, please cite our paper:


```bibtex
WORK IN PROGRESS
```

---
## ⭐ Support

If you find SpotMAP helpful, please ⭐ star our GitHub repository to support the project!


## 📧 Contact

For questions or issues, please open a [GitHub Issue](https://github.com/VikramIyr/SpotMap/issues)

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.