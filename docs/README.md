# ![SpotMAP Logo](images/spotmap_logo.png)

## Robotic Scene Understanding through Reconstruction and Dynamic Scene Graphs

**Vikram Iyer\*, Yagiz Sogancioglu\*, Roham Rahimi\*, Kerem Acar**  
*ETH Zürich*

---

## 📌 Pipeline

SpotMAP integrates:
- Robust on-board **SLAM**
- Volumetric **3D reconstruction**
- **Instance segmentation** of objects in the scene
- **Dynamic scene graph generation** to maintain a structured semantic map
- Basic **affordance estimation** for future interaction planning

<p align="center">
  <img src="images/pipeline_overview.png" alt="SpotMAP Pipeline" width="800"/>
  <br>
  <em>Figure: Overview of the SpotMAP pipeline modules.</em>
</p>

---

## 🎥 Featured Videos

- **Scene Mapping and Segmentation**  
  <p align="center">
    <img src="images/mapping_demo.gif" alt="Mapping Demo" width="600"/>
  </p>

- **Interactive Updates**  
  <p align="center">
    <img src="images/interaction_demo.gif" alt="Interaction Demo" width="600"/>
  </p>

---

## 🗂️ Dataset

SpotMAP uses RGB-D sequences recorded onboard the Boston Dynamics Spot, covering:
- Diverse indoor scenes
- Varying illumination and occlusions
- Fine-grained instance masks for common indoor objects

Each sequence includes:
- **RGB images**
- **Depth maps**
- **Pose estimates**
- **Per-frame instance masks**

---

## 🦾 Gripper Extension

To enable more reliable object interaction and grasping, a **custom 3D-printed gripper extension** is designed to attach to Spot’s end-effector.  
You can download the CAD file (`.stl`) directly from this repository.  
Need a parametric version? Feel free to contact us for the Fusion 360 source file.

<p align="center">
  <img src="images/gripper_extension.png" alt="Gripper Extension CAD" width="400"/>
</p>

---

## 📄 BibTeX

If you use SpotMAP in your work, please cite:

```bibtex
@misc{iyer2025spotmaproboticsceneunderstanding,
  title = {SpotMAP: Robotic Scene Understanding through Reconstruction and Dynamic Scene Graphs},
  author = {Vikram Iyer and Yagiz Sogancioglu and Roham Rahimi and Kerem Acar},
  year = {2025},
  eprint = {arXiv:2506.12345},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  url = {https://arxiv.org/abs/2506.12345}
}
