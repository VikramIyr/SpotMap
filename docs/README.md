<p align="center">
  <img src="images/spotmap_logo.png" alt="SpotMAP Logo">
</p>

<h2 align="center">
  Robotic Scene Understanding through Reconstruction and Dynamic Scene Graphs
</h2>

<p align="center">
  <strong>Vikram Iyer<sup>1</sup>, Yagiz Devre<sup>2</sup>, Roham Zendehdel Nobari<sup>1</sup>, Kerem Kılıç<sup>1</sup></strong><br>
  <em>ETH Zürich<sup>1</sup>, Princeton<sup>2</sup></em>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2506.12345">
    <img src="https://img.shields.io/badge/Paper-arXiv-red?style=for-the-badge&logo=arxiv" alt="Paper">
  </a>
  <a href="https://github.com/your_repo_link">
    <img src="https://img.shields.io/badge/Code-GitHub-black?style=for-the-badge&logo=github" alt="Code">
  </a>
</p>

---

## 📌 Overview

**SpotMAP** is an end-to-end framework for autonomous robotic scene understanding. It integrates:

- Time-synchronized data extraction
- Robust on-board **SLAM**
- High-fidelity **3D reconstruction**
- Precise **instance segmentation**
- **Dynamic scene graph generation** for structured semantic mapping
- Basic **affordance estimation** to support interaction planning

<p align="center">
  <img src="images/pipeline_overview.png" alt="SpotMAP Pipeline" width="800"/>
  <br>
  <em>Figure: Overview of the SpotMAP pipeline modules.</em>
</p>

---

## 🎥 Featured Demos

<p align="center">
  <iframe width="800" height="450"
          src="https://www.youtube.com/embed/ETMJrnWWVg8"
          frameborder="0"
          allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
          allowfullscreen>
  </iframe>
</p>

- **Scene Mapping & Segmentation**  
  <p align="center">
    <img src="images/mapping_demo.gif" alt="Mapping Demo" width="600"/>
  </p>

- **Interactive Updates**  
  <p align="center">
    <img src="images/interaction_demo.gif" alt="Interaction Demo" width="600"/>
  </p>

---

## 🗂️ Dataset

SpotMAP operates on RGB-D sequences recorded with the Boston Dynamics Spot robot, featuring:

- Diverse indoor environments
- Varying lighting conditions and occlusions
- Fine-grained object instance masks

Each sequence provides:

- **RGB images**
- **Depth maps**
- **Camera poses**

---

## 📄 Citation

If you find **SpotMAP** helpful in your work, please cite:

```bibtex
@misc{iyer2025spotmap,
  title = {SpotMAP: Robotic Scene Understanding through Reconstruction and Dynamic Scene Graphs},
  author = {Vikram Iyer and Yagiz Devre and Roham Zendehdel Nobari and Kerem Kılıç},
  year = {2025},
  eprint = {arXiv:2506.12345},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  url = {https://arxiv.org/abs/2506.12345}
}
