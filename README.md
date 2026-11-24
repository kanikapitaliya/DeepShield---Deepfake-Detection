# DeepShield – Intelligent Video Deepfake Detection System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.7+-orange)](https://gradio.app)
[![MSc Project](https://img.shields.io/badge/MSc-Data%20Analytics-green)]()

> **Academic Research Prototype** — MSc Data Analytics project.

## Overview

DeepShield is a video deepfake detection system that uses a **ResNeXt50 CNN + LSTM** architecture to identify manipulated facial videos. It was trained on the **FaceForensics++ (c23)** dataset and deployed as an interactive **Gradio** web application.

The system processes an uploaded video by:
1. Uniformly sampling **20 frames**
2. Detecting and cropping faces using **MTCNN**
3. Extracting deep visual features with **ResNeXt50**
4. Analysing temporal patterns with a single-layer **LSTM**
5. Producing a **REAL / FAKE** prediction with confidence score

---

## Architecture

```
Video
  │
  ├─ Frame Extraction (20 frames, uniform sampling)
  │
  ├─ Face Detection (MTCNN → crop → 224×224)
  │
  ├─ ResNeXt50 Encoder
  │    └─ conv1 → bn1 → relu → maxpool
  │       → layer1 (3 blocks) → layer2 (4 blocks)
  │       → layer3 (6 blocks) → layer4 (3 blocks)
  │       → AdaptiveAvgPool2d → [2048] feature vector per frame
  │
  ├─ LSTM (hidden=256, 1 layer, batch_first=True)
  │    └─ Processes sequence of 20 frame features
  │       Returns last hidden state → [256]
  │
  └─ Classifier
       └─ Dropout → Linear(256→128) → ReLU → Dropout → Linear(128→1)
          → sigmoid → P(FAKE) ∈ [0, 1]
```

---

## Project Structure

```
DeepShield/
├── app.py                      # Gradio application entry point
├── requirements.txt            # Python dependencies
├── README.md
│
├── models/
│   └── resnext50_lstm_best.pt  # Trained checkpoint (not in Git)
│
├── artifacts/                  # Training evaluation outputs
│   ├── test_metrics.json
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── training_history.png
│
├── src/
│   ├── __init__.py
│   ├── model.py                # Exact architecture + checkpoint loader
│   ├── video_processor.py      # Video validation and frame extraction
│   ├── face_detector.py        # MTCNN face detection and preprocessing
│   ├── predictor.py            # End-to-end inference pipeline
│   ├── explainability.py       # Grad-CAM spatial attention maps
│   └── visualizations.py       # Metrics loading and HTML formatting
│
└── tests/
    ├── test_model_loading.py
    ├── test_video_processing.py
    └── test_prediction.py
```

---

## Technologies

| Component | Library |
|-----------|---------|
| Deep Learning | PyTorch 2.1+ |
| CNN Backbone | torchvision (ResNeXt50-32x4d) |
| Video Processing | OpenCV |
| Face Detection | facenet-pytorch (MTCNN) |
| Web Interface | Gradio 4.7+ |
| Explainability | Grad-CAM (custom implementation) |

---

## Installation

### 1. Prerequisites
- Python 3.10 or higher (`py --version`)
- Git

### 2. Clone the repository
```bash
git clone https://github.com/kanikapitaliya/DeepShield---Deepfake-Detection.git
cd DeepShield---Deepfake-Detection
```

### 3. Create a virtual environment
```bash
py -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/Mac
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

> **GPU users:** Install PyTorch with CUDA support from [pytorch.org](https://pytorch.org/get-started/locally/) before running pip install.

### 5. Add the model checkpoint
Place `resnext50_lstm_best.pt` in the `models/` directory:
```
models/resnext50_lstm_best.pt
```

---

## Running the Application

```bash
# Standard launch (opens at http://127.0.0.1:7860)
py app.py

# Public shareable link (via Gradio tunnel)
py app.py --share

# Custom port
py app.py --port 8080
```

---

## Running Tests

```bash
py -m pytest tests/ -v
```

Tests that require the checkpoint are automatically skipped if the file is not present.

---

## Model & Dataset

### Model
- **Architecture:** ResNeXt50-32x4d CNN + LSTM
- **Input:** 20 frames × 224×224 RGB
- **Output:** Sigmoid probability P(FAKE)
- **Decision threshold:** 0.5

### Dataset
- **FaceForensics++** — c23 (H.264 compression) subset
- 4 manipulation methods: DeepFakes, Face2Face, FaceSwap, NeuralTextures
- 1000 original + 4000 manipulated videos

---

## Evaluation Results

> Results from the actual saved training artifacts. Not altered or fabricated.

| Metric | Value |
|--------|-------|
| Accuracy | 66.7% |
| Precision (FAKE) | 0.67 |
| Recall (FAKE) | 1.00 |
| Recall (REAL) | 0.00 |
| F1 Score (FAKE) | 0.80 |
| ROC-AUC | 0.24 |
| Test videos | 15 (10 FAKE, 5 REAL) |
| Best epoch | 1 |

### ⚠️ Important limitations

- **Small test set:** Only 15 videos were used for evaluation — results are not statistically reliable.
- **FAKE bias:** The model correctly identifies all FAKE videos but misclassifies all REAL videos as FAKE.
- **Below-chance ROC-AUC:** An AUC of 0.24 indicates the model performs below random for distinguishing REAL from FAKE in a balanced setting.
- **1 epoch of training:** The model was saved at its best checkpoint after 1 epoch, indicating significant underfitting.
- **Not production-ready:** This prototype should not be used as evidence of authenticity.

---

## Limitations & Future Work

- [ ] Train for more epochs with learning rate scheduling
- [ ] Increase training data and balance class distribution
- [ ] Evaluate on FaceForensics++ full test split
- [ ] Add data augmentation (flips, brightness, compression artifacts)
- [ ] Experiment with bi-directional LSTM or Transformer-based temporal model
- [ ] Cross-dataset evaluation (DFDC, Celeb-DF)
- [ ] Improve face detection robustness for low-quality videos

---

## Acknowledgements

- [FaceForensics++](https://github.com/ondyari/FaceForensics) dataset
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) for MTCNN implementation
- [Gradio](https://gradio.app/) for the interactive interface

---

*MSc Data Analytics · Academic Research Prototype · Not for production use*
