# ![SpotMAP Logo](images/spotmap_logo.png)

## Robotic Scene Understanding through Reconstruction and Dynamic Scene Graphs

**Vikram Iyer\*, Yagiz Devre\*, Roham Zendehdel Nobari\*, Kerem Kılıc**  
*ETH Zürich*

[📄 **Paper**](https://arxiv.org/abs/2506.12345) &nbsp; | &nbsp; [💻 **Code**](https://github.com/your_repo_link) &nbsp; | &nbsp; [🎥 **Video**](https://your_video_link)

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
