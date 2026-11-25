# DeepShield – Multimodal Deepfake Detection & Video Forensic Analysis

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.7+-orange)](https://gradio.app)
[![MSc Project](https://img.shields.io/badge/MSc-Data%20Analytics-green)]()

> **Academic Research Prototype** — MSc Data Analytics (MDA507-5), CHRIST University.

## Overview

DeepShield is a multimodal video deepfake detection system combining **visual spatiotemporal deep learning** with **physiological remote photoplethysmography (rPPG)** analysis, deployed as an interactive **Gradio** web application.

The inference pipeline processes an uploaded video through three coordinated models:
1. **Model A (Visual):** Uniformly samples **16 frames**, detects faces with **OpenCV Haar Cascade** (central 80% crop, 224x224), and extracts a 256-dim spatiotemporal embedding via **ResNeXt50-32x4d + LSTM**.
2. **Model B (Physiological):** Extracts **CHROM-rPPG** pulse signals from facial skin ROIs (requires >=8s duration and >=80% face tracking) and classifies 8 spectral/statistical features using an **ExtraTrees** classifier.
3. **Model C (Hybrid Fusion):** Fuses the 256-dim visual embedding and 8-dim rPPG vector (264 dims total) via a calibrated **LogisticRegression** pipeline (`hybrid_fusion`). When rPPG quality criteria are not met, the system automatically falls back to Model C's `visual_only` pipeline.

---

## Verified Evaluation Results (Test Set)

Results loaded directly from `artifacts/model_a/`, `artifacts/model_b/`, and `artifacts/model_c/`:

| Model / Pipeline | Accuracy | Balanced Accuracy | F1 Score | ROC-AUC | Threshold |
|------------------|----------|-------------------|----------|---------|-----------|
| **Model A** (ResNeXt50 + LSTM, n=41) | 70.7% | 69.5% | — | 0.670 | 0.51 |
| **Model B** (CHROM-rPPG ExtraTrees) | 57.6% | 52.2% | — | 0.615 | 0.70 |
| **Model C — Visual Only** | 78.8% | 76.1% | 0.857 | 0.802 | 0.52 |
| **Model C — rPPG Only** | 69.7% | 54.7% | 0.808 | 0.676 | 0.51 |
| **Model C — Hybrid Fusion** | **81.8%** | **78.0%** | **0.880** | **0.802** | 0.50 |

---

## Project Structure

```
DeepShield/
├── app.py                        # Gradio 5-tab application entry point
├── requirements.txt              # Python dependencies
├── DEVELOPMENT_PROGRESS.md       # Phase-by-phase build log
├── README.md
│
├── models/                       # Trained model files (not committed to Git)
│   ├── resnext50_lstm_v2_best.pt
│   ├── model_b_rppg_classifier.joblib
│   └── hybrid_models.joblib
│
├── artifacts/                    # Verified training/evaluation outputs
│   ├── model_a/
│   ├── model_b/
│   └── model_c/
│
├── notebooks/                    # Authoritative training & evaluation notebooks
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # Central paths, constants (16 frames, 224x224), thresholds
│   ├── model.py                  # ResNeXt50 + LSTM architecture & checkpoint loader
│   ├── video_processor.py        # Video validation and 16-frame uniform sampling
│   ├── face_detector.py          # Haar Cascade face detection & ImageNet preprocessing
│   ├── rppg.py                   # CHROM-rPPG signal extraction & quality gating
│   ├── inference_pipeline.py     # Unified 3-model pipeline with visual-only fallback
│   ├── explainability.py         # Grad-CAM spatial attention maps (encoder.7[2].conv3)
│   └── artifact_loader.py        # Loads verified metrics, reports, and plots
│
└── tests/
    ├── test_model_loading.py
    ├── test_video_processing.py
    └── test_prediction.py
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10+ (`py --version` on Windows)
- Git

### 2. Clone & Install Dependencies
```bash
git clone https://github.com/kanikapitalya/DeepShield---Deepfake-Detection.git
cd DeepShield---Deepfake-Detection
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Place Trained Models
Place the three trained model files inside `models/`:
- `models/resnext50_lstm_v2_best.pt`
- `models/model_b_rppg_classifier.joblib`
- `models/hybrid_models.joblib`

### 4. Run the Application
```bash
py app.py
# Opens at http://127.0.0.1:7860
# Optional flags: --port 8080, --share
```

### 5. Run Tests
```bash
py -m pytest tests/ -v
```

---

## Known Limitations
- **Academic Prototype:** Trained on FaceForensics++ (c23) subset; confidence scores are model probabilities, not forensic proof of authenticity.
- **rPPG Constraints:** Physiological analysis requires at least 8 seconds of video and >=80% continuous frontal face tracking; short or heavily occluded videos automatically fall back to visual-only inference.
- **Grad-CAM Scope:** Spatial attention maps reflect the ResNeXt50 encoder's final convolutional layer per frame and do not capture LSTM temporal reasoning.

---

## Team
- **Mohammed Owaiz Sherrif** — 2582425
- **Darshan Gowda** — 2582439
- **Pitaliya Kanika Shantilal** — 2582441
- CHRIST University, MDA507-5
