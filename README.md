# MediScan: Medical Image Classification System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)
[![Grad-CAM](https://img.shields.io/badge/Explainability-Grad--CAM-green.svg)](https://github.com/jacobgil/pytorch-grad-cam)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MediScan is a clinical-grade deep learning system for medical image classification and interpretability. Built using transfer learning (ResNet50 & EfficientNet-B0), the system classifies Chest X-rays for pneumonia detection, generates visual explanation heatmaps via Grad-CAM, and provides an interactive web interface for rapid diagnosis.

---

## 📅 12-Day Sprint Roadmap

- [x] **Day 1: Project Setup, Git Repository & Exploratory Data Analysis (EDA)**
- [ ] **Day 2: Data Preprocessing, Augmentation & Custom PyTorch DataLoader**
- [ ] **Day 3: Baseline Transfer Learning Architecture & Tracer Streamlit UI**
- [ ] **Day 4: Fine-Tuning ResNet50 (>85% Target) & W&B Experiment Tracking**
- [ ] **Day 5: EfficientNet-B0 Training & Architecture Comparison**
- [ ] **Day 6: Metric Evaluation (ROC-AUC, Confusion Matrix, Per-Class F1)**
- [ ] **Day 7: Grad-CAM Explainability Implementation (10+ Heatmap Cases)**
- [ ] **Day 8: Full-Featured Streamlit Interface (Single & Dual Model Predict)**
- [ ] **Day 9: Heatmap Opacity Slider, Batch Mode & CSV Export**
- [ ] **Day 10: Edge Case Analysis, Bias Detection & Failure Audit**
- [ ] **Day 11: Cloud Deployment (Streamlit Cloud) & Model Card Documentation**
- [ ] **Day 12: Final Deliverables Audit & 7-Minute Presentation Rehearsal**

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.11+ |
| **Deep Learning Framework** | PyTorch & Torchvision |
| **Backbone Models** | ResNet50 & EfficientNet-B0 |
| **Data & Augmentation** | NumPy, Pandas, Pillow, OpenCV, Albumentations |
| **Explainability** | Grad-CAM (`pytorch-grad-cam`) |
| **Web Interface** | Streamlit |
| **Experiment Tracking** | Weights & Biases (W&B) / MLflow |
| **Visualization** | Matplotlib, Seaborn |

---

## 📂 Project Structure

```text
mediscan/
├── data/
│   ├── raw/                  # Downloaded raw dataset (gitignored)
│   └── processed/            # 70/15/15 balanced splits
├── notebooks/                # Interactive experiments & prototyping
├── src/
│   ├── __init__.py
│   ├── utils.py              # Seeds, device detection, checkpointing
│   ├── dataset.py            # Dataset & augmentation loader (Day 2)
│   ├── model.py              # ResNet50 & EfficientNet definitions (Day 3 & 5)
│   ├── train.py              # Training engine & validation loop (Day 4)
│   ├── evaluate.py           # Metrics, ROC, confusion matrix (Day 6)
│   └── explainability.py     # Grad-CAM heatmap generator (Day 7)
├── app/
│   ├── app.py                # Streamlit web application (Day 8 & 9)
│   └── style.css             # Medical UI styling
├── models/                   # Saved .pt model weights
├── reports/
│   ├── figures/              # Metric plots and loss curves
│   └── gradcam_samples/      # 10+ required explainability cases
├── scripts/
│   ├── download_data.py      # Automated dataset downloader
│   └── eda.py                # Exploratory data analysis runner
├── requirements.txt
├── MODEL_CARD.md             # Formal model card documentation
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
# Clone the repository
cd "D:\OneDrive\Documents\git hub code a nova\mediscan"

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
python scripts/download_data.py
```
*(Or manually download from [Kaggle Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) into `data/raw/chest_xray/`)*

### 3. Run Exploratory Data Analysis
```bash
python scripts/eda.py
```
Outputs distribution charts and statistics directly to `reports/figures/class_distribution.png`.
