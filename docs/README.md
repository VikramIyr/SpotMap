# <p align="center">
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
  <a href="https://your_video_link">
    <img src="https://img.shields.io/badge/Video-Demo-yellow?style=for-the-badge&logo=youtube" alt="Video">
  </a>
</p>

---

## 📌 Pipeline

**SpotMAP** provides an end-to-end system for onboard robotic scene understanding. It combines:

- Time synced data extraction
- Robust **SLAM**  
- High-fidelity **3D reconstruction**
- Accurate **instance segmentation**
- **Dynamic scene graph generation** for structured semantic mapping
- Basic **affordance estimation** for planning interactions

<p align="center">
  <img src="images/pipeline_overview.png" alt="SpotMAP Pipeline" width="800"/>
  <br>
  <em>Figure: Overview of the SpotMAP pipeline modules.</em>
</p>

---

## 🎥 Featured Demos

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

SpotMAP uses RGB-D sequences captured with the Boston Dynamics Spot, covering:

- Diverse indoor scenes
- Challenging lighting and occlusions
- Fine-grained object instance masks

Each sequence includes:

- **RGB images**
- **Depth maps**
- **Camera poses**


---

## 📄 Citation

If you find SpotMAP useful in your research, please cite:

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
