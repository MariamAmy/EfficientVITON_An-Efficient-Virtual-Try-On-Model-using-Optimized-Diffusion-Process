# EfficientVITON: An Efficient Virtual Try-On Model using Optimized Diffusion Process

[![arXiv](https://img.shields.io/badge/arXiv-2501.11776-b31b1b.svg)](https://arxiv.org/abs/2501.11776)
[![Framework](https://img.shields.io/badge/PyTorch-Stable%20Diffusion-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Bachelor Thesis Project** | *Egypt-Japan University of Science and Technology (E-JUST)*

**EfficientVITON** is a high-fidelity virtual try-on framework that leverages the power of **Stable Diffusion** while optimizing for computational efficiency. By introducing a novel **Spatial Encoder** and **Zero Cross-Attention Blocks**, our model preserves fine-grained garment details and ensures realistic deformation without the heavy computational cost of traditional methods.

We also provide a **Flask-based Web Interface** to allow users to interact with the model in real-time.

---

## 🌟 Key Features

* **Optimized Diffusion Process:** Uses non-uniform timesteps to significantly reduce inference time while maintaining image quality.
* **High-Fidelity Preservation:** A dedicated Spatial Encoder ensures that complex clothing textures and patterns are preserved.
* **Robust to Occlusions:** Effectively handles complex poses and self-occlusions using DensePose and Semantic Segmentation (LIP).
* **Interactive Web Platform:** A user-friendly Flask web app for uploading images and viewing results instantly.

---

## 🏗️ System Architecture

Our pipeline consists of three main stages:

1.  **Preprocessing:** Extracts human parsing maps (LIP), pose estimation (OpenPose), and dense pose (DensePose) to create a "clothing-agnostic" representation.
2.  **Feature Encoding:** The **Spatial Encoder** captures high-frequency details from the garment image.
3.  **Diffusion Generation:** The **Main U-Net** (initialized from Stable Diffusion) synthesizes the final image, guided by **Zero Cross-Attention** blocks that fuse garment features into the generation process.

---

## 📂 Project Structure

```text
├── Code/
│   ├── Preprocessing/      
│   │   ├── preprocessing/  # Scripts for OpenPose, DensePose, and Parsing
│   │   └── ...
│   ├── Models/             # Definition of Spatial Encoder and U-Net
│   ├── app.py              # Flask Web Application entry point
│   └── main.py             # Main script for Training and Inference
├── Configs/                # YAML configuration files
├── Data/                   # Dataset folder (VITON-HD / DressCode)
├── checkpoints/            # Pre-trained models (final.pth)
├── requirements.txt        # Python dependencies
└── README.md
```

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/MariamAmy/EfficientVITON_An-Efficient-Virtual-Try-On-Model-using-Optimized-Diffusion-Process.git](https://github.com/MariamAmy/EfficientVITON_An-Efficient-Virtual-Try-On-Model-using-Optimized-Diffusion-Process.git)
cd EfficientVITON_An-Efficient-Virtual-Try-On-Model-using-Optimized-Diffusion-Process
```
### 2. Install Dependencies
```bash
conda create -n efficientviton python=3.8
conda activate efficientviton
pip install -r requirements.txt
```

## Citation
```bash
@article{mahmoud2025efficientviton,
  title={EfficientVITON: An Efficient Virtual Try-On Model using Optimized Diffusion Process},
  author={Mahmoud, Mariam A. M. and Atef, Mostafa and Rashed, Ahmed and et al.},
  journal={arXiv preprint arXiv:2501.11776},
  year={2025}
}
```

## 🤝 Acknowledgements
This project uses code and models from:

- Stable Diffusion
- OpenPose
- Detectron2
- Schp (Self-Correction Human Parsing)

We thank the authors for their open-source contributions.
