# DeepShield — Development Progress

## Project Overview
Multimodal Deepfake Detection and Video Forensic Analysis system for MSc Data Analytics (MDA507-5, CHRIST University).
Combines visual spatiotemporal analysis (Model A), physiological rPPG signal analysis (Model B), and late hybrid fusion (Model C).

---

## Completed Phases

### Phase 1: Repository Audit & Notebook Inspection
- Audited all four training notebooks in `notebooks/`.
- Verified Model A checkpoint `models/resnext50_lstm_v2_best.pt`:
  - `ResNeXt50_32x4d` encoder + single-layer `LSTM(2048 -> 256)` + `MLP(256 -> 128 -> 1)`
  - `NUM_FRAMES = 16`, `IMAGE_SIZE = 224`, Haar Cascade face detection (central 80% crop)
- Verified Model B (`models/model_b_rppg_classifier.joblib`):
  - `ExtraTreesClassifier`, 8 CHROM-rPPG features, threshold `0.70`
- Verified Model C (`models/hybrid_models.joblib`):
  - `SimpleImputer -> StandardScaler -> LogisticRegression` pipelines for `hybrid_fusion` (264 dims, threshold `0.50`), `visual_only` (256 dims, threshold `0.52`), and `rppg_only` (8 dims, threshold `0.51`).
- Verified evaluation artifacts in `artifacts/model_a/`, `artifacts/model_b/`, `artifacts/model_c/`.

### Phase 2: Model A Backend & Configuration
- Created `src/config.py` with centralized paths, `NUM_FRAMES=16`, `IMAGE_SIZE=224`, and model thresholds.
- Updated `src/model.py` (`DeepFakeDetector` with `forward_embedding()` returning 256-dim LSTM representation; strict checkpoint loading).
- Updated `src/video_processor.py` for 16-frame uniform sampling and validation.
- Updated `src/face_detector.py` to use OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`) with central 80% crop and ImageNet normalization.

### Phase 3: Model B Physiological rPPG Pipeline
- Created `src/rppg.py` implementing the CHROM-rPPG pipeline:
  - <=30 FPS sampling, Haar face tracking (>=80% track ratio), central ROI (20-80% x, 20-75% y)
  - HSV skin mask (`S>20`, `S<220`, `V>40`), windowed CHROM, Butterworth bandpass (`0.7-3.0 Hz`), Welch PSD
  - Extracts 8 features in exact training order; rejects videos <8s (`video_too_short`) or with insufficient face tracking (`insufficient_face_track`).

### Phase 4: Model C Hybrid Fusion & Unified Pipeline
- Created `src/inference_pipeline.py` (`InferencePipeline`, `InferenceResult`):
  - Runs Model A -> 256-dim embedding + visual score
  - Runs Model B -> 8 rPPG features + physiological score (with quality rejection)
  - Runs Model C `hybrid_fusion` (264 dims) when rPPG succeeds, or `visual_only` (256 dims) fallback when rPPG is unavailable
  - Includes `SimpleImputer` compatibility shim for scikit-learn 1.6.1 -> 1.9.x unpickling.
- Created `src/artifact_loader.py` to load verified JSON/TXT/PNG evaluation artifacts from `artifacts/model_a/`, `model_b/`, and `model_c/`.

### Phase 5: Gradio Frontend & Cleanup
- Redesigned `app.py` with a dark GitHub-inspired theme and 5 tabs:
  1. **Video Analysis** — upload, 3-model verdict, routing indicator, confidence breakdown
  2. **Visual Forensics** — sampled face crops and Grad-CAM spatial attention maps (`src/explainability.py`)
  3. **Physiological Analysis** — CHROM-rPPG signal metrics and quality status
  4. **Model Performance** — verified metrics, confusion matrices, ROC curves, and comparison charts
  5. **Architecture** — multimodal pipeline overview and team details
- Removed obsolete `src/predictor.py` and `src/visualizations.py`.
- Updated `.gitignore` to exclude `sample_videos/`, `*.joblib`, and `.pt` checkpoints.
- Updated `requirements.txt` (`opencv-python>=4.8.0,<5.0.0`, `joblib`, `scikit-learn`).

### Phase 6: Testing & Verification
- Verified OpenCV 4.14.0.94 (`cv2.CascadeClassifier`) and all imports.
- Ran end-to-end `InferencePipeline` on all 4 sample videos:
  - `sample_videos/097_033.mp4`: `FAKE (0.990)` via `hybrid_fusion` (`rppg=ok`, `16/16` faces, `20.4s`)
  - `sample_videos/147_055.mp4`: `FAKE (0.998)` via `hybrid_fusion` (`rppg=ok`, `16/16` faces, `15.2s`)
  - `sample_videos/sample_data.mp4`: `FAKE (0.546)` via `visual_only` (`insufficient_face_track`, `14/16` faces, `10.4s`)
  - `sample_videos/vd.mp4`: `FAKE (0.966)` via `visual_only` (`video_too_short`, `16/16` faces, `2.7s`)
- Verified `app.build_interface()` builds cleanly.
- Updated `tests/test_model_loading.py`, `tests/test_video_processing.py`, and `tests/test_prediction.py` for the 16-frame multimodal pipeline.
- Ran `py -m pytest tests/ -v`: **27 passed** in 21.48s.

---

## Pending Tasks
- Commit and push all verified changes to `origin/main`.
