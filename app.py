"""
app.py
======
DeepShield – Intelligent Video Deepfake Detection System
Gradio frontend application.

Tabs:
  1. Home / Overview
  2. Video Detection
  3. Frame Analysis
  4. Model Performance
  5. About the Model

Usage:
  python app.py
  python app.py --share          # public shareable link
  python app.py --port 7860
"""

import gradio as gr
import torch
import numpy as np
import logging
import os
import sys
import argparse
import tempfile
from PIL import Image
from typing import Optional

# Configure logging before importing src modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Startup: load model once
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "resnext50_lstm_best.pt")

from src.model import load_model, get_device
from src.predictor import run_inference, PredictionResult, VideoValidationError
from src.visualizations import (
    load_test_metrics,
    load_classification_report,
    load_image_artifact,
    format_metrics_html,
    format_prediction_html,
)
from src.explainability import try_gradcam

device = get_device()
_model = None


def get_model():
    """Lazy singleton model loader."""
    global _model
    if _model is None:
        logger.info("Loading model for the first time...")
        _model = load_model(MODEL_PATH, device)
    return _model


# ---------------------------------------------------------------------------
# Detection handler
# ---------------------------------------------------------------------------
def detect_video(video_file):
    """
    Main inference handler called by the Gradio interface.

    Args:
        video_file: file path string provided by Gradio file upload

    Returns:
        tuple of (prediction_html, status_text, face_crops_gallery, gradcam_gallery, warning_text)
    """
    if video_file is None:
        return (
            "<p style='color:#94a3b8; padding:1rem;'>Upload a video to begin analysis.</p>",
            "",
            [],
            [],
            "",
        )

    try:
        model = get_model()

        # Run full pipeline
        result: PredictionResult = run_inference(
            video_path=video_file,
            model=model,
            device=device,
        )

        # Prediction card HTML
        pred_html = format_prediction_html(
            result.label, result.confidence, result.warning
        )

        # Status summary text
        status = (
            f"✓ Analyzed {result.total_frames} frames · "
            f"Faces in {result.faces_detected}/{result.total_frames} frames · "
            f"Duration: {result.video_meta.get('duration_sec', 0):.1f}s · "
            f"Time: {result.inference_time_sec:.1f}s"
        )

        # Face crop gallery (PIL Images)
        face_pil = [Image.fromarray(c) for c in result.face_crops] if result.face_crops else []

        # Grad-CAM (best-effort)
        gradcam_maps = []
        try:
            # Reconstruct tensor for Grad-CAM
            from src.video_processor import extract_frames
            from src.face_detector import preprocess_frames
            frames_bgr, _ = extract_frames(video_file, num_frames=20)
            tensor, _ = preprocess_frames(frames_bgr)
            raw_maps = try_gradcam(model, tensor, device, max_maps=6)
            gradcam_maps = [Image.fromarray(m) for m in raw_maps]
        except Exception as e:
            logger.warning(f"Grad-CAM skipped: {e}")

        warning_text = result.warning or ""

        return pred_html, status, face_pil, gradcam_maps, warning_text

    except VideoValidationError as e:
        err_html = f"""
        <div style="background:#450a0a; border:1px solid #ef4444; border-radius:8px;
                    padding:1rem; color:#fca5a5; font-family:monospace;">
          ❌ Video Error: {str(e)}
        </div>"""
        return err_html, "Validation failed.", [], [], str(e)

    except Exception as e:
        logger.exception("Unexpected error during inference")
        err_html = f"""
        <div style="background:#450a0a; border:1px solid #ef4444; border-radius:8px;
                    padding:1rem; color:#fca5a5;">
          ❌ An unexpected error occurred. Please try another video.<br>
          <small style='opacity:0.7'>Check application logs for details.</small>
        </div>"""
        return err_html, "Error during analysis.", [], [], ""


# ---------------------------------------------------------------------------
# Performance dashboard data (loaded once)
# ---------------------------------------------------------------------------
_metrics = load_test_metrics()
_report  = load_classification_report()
_cm_img  = load_image_artifact("confusion_matrix.png")
_roc_img = load_image_artifact("roc_curve.png")
_hist_img = load_image_artifact("training_history.png")

metrics_html = format_metrics_html(_metrics)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* Global */
.gradio-container {
    font-family: 'Segoe UI', system-ui, sans-serif !important;
    background: #0f172a !important;
    color: #f1f5f9 !important;
}

/* Tabs */
.tab-nav button {
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.5rem 1rem !important;
}
.tab-nav button.selected {
    color: #3b82f6 !important;
    border-bottom-color: #3b82f6 !important;
}

/* Upload box */
.upload-box {
    border: 2px dashed #334155 !important;
    border-radius: 12px !important;
    background: #1e293b !important;
}

/* Buttons */
.btn-analyze {
    background: #3b82f6 !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
}

/* Gallery */
.gallery-item img {
    border-radius: 8px !important;
    border: 1px solid #334155 !important;
}

/* Code / report blocks */
.report-block {
    background: #1e293b;
    border-radius: 8px;
    padding: 1rem;
    font-family: monospace;
    font-size: 0.85rem;
    color: #e2e8f0;
    white-space: pre;
    overflow-x: auto;
}
"""

# ---------------------------------------------------------------------------
# About content
# ---------------------------------------------------------------------------
ABOUT_HTML = """
<div style="font-family:'Segoe UI',sans-serif; color:#f1f5f9; padding:0.5rem;">
  <h2 style="color:#3b82f6;">🧠 About the DeepShield Model</h2>

  <h3 style="color:#e2e8f0; margin-top:1.5rem;">Architecture</h3>
  <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:1rem; margin:1rem 0;">
    <div style="background:#1e293b; border-radius:10px; padding:1rem; border-top:3px solid #3b82f6;">
      <div style="font-size:1.5rem;">🏗️</div>
      <strong style="color:#3b82f6;">ResNeXt50 CNN</strong>
      <p style="color:#94a3b8; font-size:0.85rem; margin-top:0.5rem;">
        Extracts rich spatial and visual features from each individual video frame.
        Pre-trained on ImageNet for robust visual representations.
      </p>
    </div>
    <div style="background:#1e293b; border-radius:10px; padding:1rem; border-top:3px solid #8b5cf6;">
      <div style="font-size:1.5rem;">⏱️</div>
      <strong style="color:#8b5cf6;">LSTM</strong>
      <p style="color:#94a3b8; font-size:0.85rem; margin-top:0.5rem;">
        Processes the sequence of frame features across time. Learns temporal
        inconsistencies introduced by deepfake generation methods.
      </p>
    </div>
    <div style="background:#1e293b; border-radius:10px; padding:1rem; border-top:3px solid #10b981;">
      <div style="font-size:1.5rem;">🎯</div>
      <strong style="color:#10b981;">Classifier</strong>
      <p style="color:#94a3b8; font-size:0.85rem; margin-top:0.5rem;">
        A two-layer MLP that takes the LSTM's final hidden state and outputs
        a single probability: P(FAKE) ∈ [0, 1].
      </p>
    </div>
  </div>

  <h3 style="color:#e2e8f0; margin-top:1.5rem;">Dataset</h3>
  <div style="background:#1e293b; border-radius:10px; padding:1rem; margin:0.5rem 0;">
    <p style="color:#cbd5e1; margin:0;">
      Trained on the <strong style="color:#f1f5f9;">FaceForensics++</strong> dataset (c23 compression subset).
      FaceForensics++ contains videos manipulated by four deepfake methods: DeepFakes, Face2Face,
      FaceSwap, and NeuralTextures, alongside pristine source videos.
    </p>
  </div>

  <h3 style="color:#e2e8f0; margin-top:1.5rem;">⚠️ Known Limitations</h3>
  <ul style="color:#94a3b8; font-size:0.9rem; line-height:1.8;">
    <li><strong style="color:#f1f5f9;">Small test set:</strong> Evaluated on only 15 videos (10 FAKE, 5 REAL) — not statistically reliable.</li>
    <li><strong style="color:#f1f5f9;">FAKE bias:</strong> The model currently predicts FAKE for most inputs due to class imbalance and limited training.</li>
    <li><strong style="color:#f1f5f9;">1 epoch training:</strong> The model was saved at its best epoch (epoch 1), indicating underfitting.</li>
    <li><strong style="color:#f1f5f9;">Dataset bias:</strong> May not generalize to manipulation methods not present in FaceForensics++.</li>
    <li><strong style="color:#f1f5f9;">Compression sensitivity:</strong> Performance degrades on highly compressed or low-resolution videos.</li>
    <li><strong style="color:#f1f5f9;">Not production-ready:</strong> This is a research prototype for academic purposes only.</li>
  </ul>

  <h3 style="color:#e2e8f0; margin-top:1.5rem;">Technical Stack</h3>
  <div style="display:flex; gap:0.75rem; flex-wrap:wrap;">
    <span style="background:#1e293b; color:#3b82f6; border:1px solid #3b82f6; border-radius:999px; padding:0.25rem 0.75rem; font-size:0.82rem;">PyTorch</span>
    <span style="background:#1e293b; color:#8b5cf6; border:1px solid #8b5cf6; border-radius:999px; padding:0.25rem 0.75rem; font-size:0.82rem;">torchvision</span>
    <span style="background:#1e293b; color:#10b981; border:1px solid #10b981; border-radius:999px; padding:0.25rem 0.75rem; font-size:0.82rem;">OpenCV</span>
    <span style="background:#1e293b; color:#f59e0b; border:1px solid #f59e0b; border-radius:999px; padding:0.25rem 0.75rem; font-size:0.82rem;">Gradio</span>
    <span style="background:#1e293b; color:#ef4444; border:1px solid #ef4444; border-radius:999px; padding:0.25rem 0.75rem; font-size:0.82rem;">facenet-pytorch</span>
    <span style="background:#1e293b; color:#06b6d4; border:1px solid #06b6d4; border-radius:999px; padding:0.25rem 0.75rem; font-size:0.82rem;">Python 3.12</span>
  </div>

  <div style="background:#0f172a; border:1px solid #334155; border-radius:10px; padding:1rem; margin-top:1.5rem;">
    <strong style="color:#94a3b8; font-size:0.82rem;">
      MSc Data Analytics · DeepShield Project · Academic Research Prototype
    </strong>
  </div>
</div>
"""

HOME_HTML = """
<div style="font-family:'Segoe UI',sans-serif; color:#f1f5f9; padding:0.5rem;">
  <div style="text-align:center; padding:2rem 0 1.5rem 0;">
    <div style="font-size:3rem; margin-bottom:0.5rem;">🛡️</div>
    <h1 style="font-size:2.2rem; font-weight:800; color:#3b82f6; margin:0;">DeepShield</h1>
    <p style="color:#94a3b8; font-size:1.05rem; margin-top:0.5rem;">
      Intelligent Video Deepfake Detection · MSc Data Analytics Project
    </p>
  </div>

  <div style="max-width:720px; margin:0 auto;">
    <div style="background:#1e293b; border-radius:12px; padding:1.25rem; margin-bottom:1rem;">
      <h3 style="color:#e2e8f0; margin:0 0 0.75rem 0;">What is a deepfake?</h3>
      <p style="color:#94a3b8; font-size:0.9rem; margin:0; line-height:1.7;">
        Deepfakes are synthetically manipulated videos in which a person's face or voice
        is replaced or altered using deep learning generative models. They pose serious risks
        to media authenticity, public trust, and personal privacy.
      </p>
    </div>

    <div style="background:#1e293b; border-radius:12px; padding:1.25rem; margin-bottom:1rem;">
      <h3 style="color:#e2e8f0; margin:0 0 0.75rem 0;">How DeepShield works</h3>
      <div style="display:flex; flex-direction:column; gap:0.5rem;">
        <div style="display:flex; align-items:flex-start; gap:0.75rem; color:#94a3b8; font-size:0.88rem;">
          <span style="color:#3b82f6; font-weight:700; min-width:1.5rem;">1.</span>
          <div><strong style="color:#e2e8f0;">Upload Video</strong> — Accepts MP4, AVI, MOV, MKV files up to 500 MB.</div>
        </div>
        <div style="display:flex; align-items:flex-start; gap:0.75rem; color:#94a3b8; font-size:0.88rem;">
          <span style="color:#3b82f6; font-weight:700; min-width:1.5rem;">2.</span>
          <div><strong style="color:#e2e8f0;">Extract Frames</strong> — 20 frames are uniformly sampled from the video.</div>
        </div>
        <div style="display:flex; align-items:flex-start; gap:0.75rem; color:#94a3b8; font-size:0.88rem;">
          <span style="color:#3b82f6; font-weight:700; min-width:1.5rem;">3.</span>
          <div><strong style="color:#e2e8f0;">Detect Faces</strong> — MTCNN locates and crops the face region in each frame.</div>
        </div>
        <div style="display:flex; align-items:flex-start; gap:0.75rem; color:#94a3b8; font-size:0.88rem;">
          <span style="color:#3b82f6; font-weight:700; min-width:1.5rem;">4.</span>
          <div><strong style="color:#e2e8f0;">CNN Features</strong> — ResNeXt50 extracts deep visual features from each face crop.</div>
        </div>
        <div style="display:flex; align-items:flex-start; gap:0.75rem; color:#94a3b8; font-size:0.88rem;">
          <span style="color:#3b82f6; font-weight:700; min-width:1.5rem;">5.</span>
          <div><strong style="color:#e2e8f0;">Temporal Analysis</strong> — LSTM processes the frame sequence to detect inconsistencies.</div>
        </div>
        <div style="display:flex; align-items:flex-start; gap:0.75rem; color:#94a3b8; font-size:0.88rem;">
          <span style="color:#3b82f6; font-weight:700; min-width:1.5rem;">6.</span>
          <div><strong style="color:#e2e8f0;">Predict</strong> — The classifier outputs REAL or FAKE with a confidence score.</div>
        </div>
      </div>
    </div>

    <div style="background:#172554; border:1px solid #1d4ed8; border-radius:12px; padding:1rem;">
      <p style="color:#93c5fd; font-size:0.82rem; margin:0;">
        ⚠️ <strong>Academic disclaimer:</strong> This system is a research prototype built for an MSc project.
        It is not production-ready. The model's current evaluation metrics reflect the limitations of a small
        training run and should not be used as sole evidence of video authenticity.
      </p>
    </div>
  </div>
</div>
"""

FRAME_ANALYSIS_NOTE = """
**ℹ️ Frame Analysis Note**

The DeepShield model produces a **video-level prediction** via the LSTM, not per-frame scores.
The face crops shown below are the actual inputs fed to the ResNeXt50 encoder.

**Grad-CAM heatmaps** (if shown) indicate spatial attention of the CNN's last convolutional layer —
this is exploratory and does not directly reflect per-frame deepfake likelihood.
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
)

# ---------------------------------------------------------------------------
# Build Gradio interface
# ---------------------------------------------------------------------------
def build_interface():
    with gr.Blocks(title="DeepShield – Deepfake Detection") as demo:
        # ---- Header ----
        gr.HTML("""
        <div style="background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);
                    padding:1rem 1.5rem; display:flex; align-items:center; gap:0.75rem;
                    border-bottom:1px solid #1e293b;">
          <span style="font-size:1.6rem;">🛡️</span>
          <div>
            <div style="color:#3b82f6; font-size:1.1rem; font-weight:700;">DeepShield</div>
            <div style="color:#64748b; font-size:0.75rem;">Intelligent Video Deepfake Detection · MSc Data Analytics</div>
          </div>
        </div>
        """)

        with gr.Tabs():

            # ================================================================
            # TAB 1: Home
            # ================================================================
            with gr.Tab("🏠 Home"):
                gr.HTML(HOME_HTML)

            # ================================================================
            # TAB 2: Video Detection
            # ================================================================
            with gr.Tab("🔍 Video Detection"):
                gr.Markdown("### Upload a video to analyse")
                gr.Markdown(
                    "Supported formats: MP4, AVI, MOV, MKV, WebM · Max size: 500 MB"
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(
                            label="Upload Video",
                            sources=["upload"],
                        )
                        analyze_btn = gr.Button(
                            "🔍 Analyse Video",
                            variant="primary",
                            size="lg",
                        )
                        status_text = gr.Textbox(
                            label="Analysis Status",
                            interactive=False,
                            show_label=True,
                            lines=2,
                        )

                    with gr.Column(scale=1):
                        prediction_output = gr.HTML(
                            value="<p style='color:#64748b; padding:1.5rem;'>"
                                  "Upload a video and click Analyse.</p>",
                            label="Prediction",
                        )
                        warning_output = gr.Textbox(
                            label="⚠️ Warnings",
                            interactive=False,
                            visible=True,
                            lines=2,
                        )

                # State to hold face crops and gradcam for Frame Analysis tab
                face_crops_state = gr.State([])
                gradcam_state    = gr.State([])

                analyze_btn.click(
                    fn=detect_video,
                    inputs=[video_input],
                    outputs=[
                        prediction_output,
                        status_text,
                        face_crops_state,
                        gradcam_state,
                        warning_output,
                    ],
                )

            # ================================================================
            # TAB 3: Frame Analysis
            # ================================================================
            with gr.Tab("🖼️ Frame Analysis"):
                gr.Markdown(FRAME_ANALYSIS_NOTE)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Face Crops (model inputs)")
                        face_gallery = gr.Gallery(
                            label="Extracted Face Crops",
                            columns=4,
                            rows=2,
                            object_fit="contain",
                            height=300,
                        )

                    with gr.Column():
                        gr.Markdown("#### Grad-CAM Heatmaps (CNN attention)")
                        gradcam_gallery = gr.Gallery(
                            label="Grad-CAM Spatial Attention",
                            columns=4,
                            rows=2,
                            object_fit="contain",
                            height=300,
                        )

                gr.Markdown(
                    "> **Prediction is video-level** (LSTM processes all 20 frames together). "
                    "No per-frame deepfake probability is available from this architecture."
                )

                # Populate galleries when detection is run
                analyze_btn_ref_note = gr.HTML(
                    "<p style='color:#64748b;font-size:0.85rem;'>"
                    "Run detection from the <b>Video Detection</b> tab to populate these galleries.</p>"
                )

                # Wire up the state → galleries via a secondary event
                # We use a dummy button on this tab to refresh from state
                refresh_btn = gr.Button("🔄 Refresh from last analysis", size="sm")

                def refresh_galleries(faces, gradcams):
                    return faces or [], gradcams or []

                refresh_btn.click(
                    fn=refresh_galleries,
                    inputs=[face_crops_state, gradcam_state],
                    outputs=[face_gallery, gradcam_gallery],
                )

                # Also update automatically when detect_video completes
                analyze_btn.click(
                    fn=refresh_galleries,
                    inputs=[face_crops_state, gradcam_state],
                    outputs=[face_gallery, gradcam_gallery],
                )

            # ================================================================
            # TAB 4: Model Performance
            # ================================================================
            with gr.Tab("📊 Model Performance"):
                gr.HTML(metrics_html)

                with gr.Row():
                    if _cm_img:
                        gr.Image(
                            value=_cm_img,
                            label="Confusion Matrix",
                            show_label=True,
                            interactive=False,
                        )
                    else:
                        gr.Markdown("*Confusion matrix not found in artifacts/*")

                    if _roc_img:
                        gr.Image(
                            value=_roc_img,
                            label="ROC Curve",
                            show_label=True,
                            interactive=False,
                        )
                    else:
                        gr.Markdown("*ROC curve not found in artifacts/*")

                if _hist_img:
                    gr.Image(
                        value=_hist_img,
                        label="Training History",
                        show_label=True,
                        interactive=False,
                    )
                else:
                    gr.Markdown("*Training history not found in artifacts/*")

                gr.Markdown("#### Classification Report")
                gr.Code(
                    value=_report,
                    language=None,
                    label="Per-Class Metrics",
                    interactive=False,
                )

            # ================================================================
            # TAB 5: About
            # ================================================================
            with gr.Tab("ℹ️ About"):
                gr.HTML(ABOUT_HTML)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepShield Deepfake Detection")
    parser.add_argument("--share", action="store_true", help="Create public shareable link")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("DeepShield – Starting up")
    logger.info(f"Device: {device}")
    logger.info(f"Model checkpoint: {MODEL_PATH}")
    logger.info(f"Port: {args.port}")
    logger.info("=" * 50)

    # Pre-load the model at startup
    try:
        get_model()
        logger.info("Model loaded successfully at startup.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Application will start but predictions will fail until model is available.")

    demo = build_interface()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=False,          # Don't expose stack traces in UI
        favicon_path=None,
        theme=THEME,
        css=CUSTOM_CSS,
    )
