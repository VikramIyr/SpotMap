# SpotMAP

## 📌 Overview

**SpotMAP** is a complete end-to-end pipeline for robotic scene understanding and interactive mapping. It integrates on-board SLAM, volumetric 3D reconstruction, instance segmentation, dynamic scene graph generation, and affordance prediction — giving mobile robots the capability to build detailed maps and understand their surroundings with minimal human supervision.

---

## 📂 Repository Structure

```plaintext
SpotMAP/
├── src/          # Source code: models, datasets, pipeline 
├── configs/      # Configuration files 
├── scripts/      # Shell scripts 
├── data/         
├── docs/         # Additional documentation and figures
```
### 📂 Source code
```plaintext
src/
├── /           
├── /       
├── /      
├── /         
├── /         
```
---

## 🚀 Setup


### OpenMask3d Docker

SpotMAP relies on the OpenMask3D Docker container to ensure a consistent and reproducible runtime environment.

#### 1️⃣ Pull the image and run the container
```bash
docker pull craiden/openmask:v1.0
docker run -p 5001:5001 --gpus all -it craiden/openmask:v1.0
```

#### 2️⃣ Follow the next steps

For detailed instructions on how to prepare your data, configure the container, and run OpenMask3D, please refer to the [OpenMask3D repository](https://github.com/OpenMask3D/openmask3d) and its setup guide.




## 📄 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{YOUR_BIBTEX_KEY,
  title     = {$PAPER_TITLE},
  author    = {First Author and Second Author and Third Author},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {$CVPR_YEAR}
}
```

---

## 📧 Contact

For questions or issues, please open a [GitHub Issue](https://github.com/VikramIyr/SpotMap/issues)

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.