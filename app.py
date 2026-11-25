"""
app.py
======
DeepShield – Multimodal Deepfake Detection and Video Forensic Analysis
Gradio 6 frontend — professional dark theme.

Sections:
  1. Video Analysis      — upload, run all 3 models, show verdict
  2. Visual Forensics    — face crops, Grad-CAM heatmaps
  3. Physiological       — rPPG signal features (if available)
  4. Model Performance   — verified evaluation artifacts
  5. Architecture        — pipeline description and team info

Usage:
    py app.py
    py app.py --port 7861
    py app.py --share
"""

import argparse
import logging
import os
import sys
import time

import gradio as gr
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global pipeline (loaded once at startup)
# ---------------------------------------------------------------------------
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from src.inference_pipeline import InferencePipeline
        _pipeline = InferencePipeline()
    return _pipeline


# ---------------------------------------------------------------------------
# Load artifacts once at import time
# ---------------------------------------------------------------------------
from src.artifact_loader import (
    load_model_a_metrics, load_model_a_report,
    load_model_a_confusion_matrix, load_model_a_roc_curve, load_model_a_training_history,
    load_model_b_metrics, load_model_b_report,
    load_model_b_confusion_matrix, load_model_b_roc_curve, load_model_b_feature_importance,
    load_model_c_metrics, load_model_c_comparison_chart,
    load_model_c_hybrid_confusion_matrix, load_model_c_visual_confusion_matrix,
    load_model_c_rppg_confusion_matrix,
)

_a_metrics    = load_model_a_metrics()
_a_report     = load_model_a_report()
_a_cm         = load_model_a_confusion_matrix()
_a_roc        = load_model_a_roc_curve()
_a_hist       = load_model_a_training_history()

_b_metrics    = load_model_b_metrics()
_b_report     = load_model_b_report()
_b_cm         = load_model_b_confusion_matrix()
_b_roc        = load_model_b_roc_curve()
_b_fi         = load_model_b_feature_importance()

_c_metrics    = load_model_c_metrics()
_c_comp       = load_model_c_comparison_chart()
_c_hybrid_cm  = load_model_c_hybrid_confusion_matrix()
_c_visual_cm  = load_model_c_visual_confusion_matrix()
_c_rppg_cm    = load_model_c_rppg_confusion_matrix()


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CSS = """
/* ── Global ── */
.gradio-container {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
    background: #0d1117 !important;
    color: #e6edf3 !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
}

/* ── Tabs ── */
.tab-nav { border-bottom: 1px solid #21262d !important; }
.tab-nav button {
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: #8b949e !important;
    padding: 0.6rem 1.2rem !important;
    border-bottom: 2px solid transparent !important;
}
.tab-nav button.selected {
    color: #58a6ff !important;
    border-bottom-color: #58a6ff !important;
    background: transparent !important;
}

/* ── Cards ── */
.card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}
.card-blue  { border-top: 3px solid #58a6ff; }
.card-teal  { border-top: 3px solid #39d353; }
.card-amber { border-top: 3px solid #d29922; }
.card-red   { border-top: 3px solid #f85149; }

/* ── Verdict banner ── */
.verdict-fake {
    background: #3d1f1f !important;
    border: 2px solid #f85149 !important;
    border-radius: 12px !important;
    text-align: center;
}
.verdict-real {
    background: #1a2d1a !important;
    border: 2px solid #39d353 !important;
    border-radius: 12px !important;
    text-align: center;
}

/* ── Misc ── */
label { color: #8b949e !important; font-size: 0.82rem !important; }
.upload-box { border: 2px dashed #30363d !important; background: #161b22 !important; border-radius: 10px !important; }
"""

# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _metric_card(title: str, value: str, color: str = "#58a6ff") -> str:
    return f"""
    <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;
                padding:0.9rem 1rem;text-align:center;min-width:120px;">
      <div style="color:#8b949e;font-size:0.75rem;text-transform:uppercase;letter-spacing:.05em;">{title}</div>
      <div style="color:{color};font-size:1.5rem;font-weight:700;margin-top:0.25rem;">{value}</div>
    </div>"""


def build_verdict_html(result) -> str:
    """Build the main verdict card HTML from an InferenceResult."""
    color      = "#f85149" if result.is_fake else "#39d353"
    cls        = "verdict-fake" if result.is_fake else "verdict-real"
    pct        = _fmt_pct(result.confidence)
    route_map  = {
        "hybrid_fusion": "Hybrid (Visual + Physiological)",
        "visual_only":   "Visual Only (Physiological unavailable)",
        "error":         "Error",
    }
    route_label = route_map.get(result.route, result.route)

    # Route-specific secondary info
    sub_lines = []
    if result.route == "hybrid_fusion":
        sub_lines.append(
            f"Visual score: {_fmt_pct(result.visual_prob)} &nbsp;|&nbsp; "
            f"Physiological score: {_fmt_pct(result.rppg_prob)}"
        )
    else:
        sub_lines.append(f"Visual score: {_fmt_pct(result.visual_prob)}")

    sub_lines.append(f"Analysis route: <strong>{route_label}</strong>")

    sub_html = "<br>".join(
        f"<span style='color:#8b949e;font-size:0.85rem;'>{s}</span>"
        for s in sub_lines
    )

    # Warning
    warn_html = ""
    if result.warning:
        warn_html = f"""
        <div style="background:#261a00;border:1px solid #d29922;border-radius:8px;
                    padding:0.75rem 1rem;margin-top:1rem;text-align:left;">
          <span style="color:#d29922;font-size:0.82rem;">
            {result.warning}
          </span>
        </div>"""

    # Disclaimer
    disclaimer = """
    <div style="margin-top:1rem;padding:0.6rem 1rem;background:#0d1117;
                border-radius:6px;border:1px solid #21262d;">
      <span style="color:#6e7681;font-size:0.78rem;">
        Confidence is a model score, not cryptographic proof of authenticity.
        This system is a research prototype and may produce false positives/negatives.
      </span>
    </div>"""

    return f"""
    <div class="{cls}" style="padding:1.5rem 1rem;border-radius:12px;">
      <div style="font-size:2.5rem;font-weight:800;color:{color};letter-spacing:.02em;">
        {result.label}
      </div>
      <div style="font-size:1.2rem;color:{color};margin-top:0.25rem;">
        Confidence: {pct}
      </div>
      <div style="margin-top:0.75rem;">{sub_html}</div>
    </div>
    {warn_html}
    {disclaimer}
    """


def build_status_html(result) -> str:
    duration  = result.video_meta.get('duration_sec', 0)
    fps       = result.video_meta.get('fps', 0)
    res       = f"{result.video_meta.get('width', '?')}×{result.video_meta.get('height', '?')}"
    return (
        f"Analyzed {result.total_frames} frames | "
        f"Faces: {result.faces_detected}/{result.total_frames} | "
        f"Duration: {duration:.1f}s | FPS: {fps:.1f} | "
        f"Resolution: {res} | "
        f"Time: {result.inference_time_sec:.1f}s"
    )


def build_rppg_html(result) -> str:
    if not result.rppg_available:
        reason = result.rppg_status.replace("_", " ")
        return f"""
        <div style="background:#161b22;border:1px solid #21262d;border-radius:10px;
                    padding:1.2rem;color:#8b949e;text-align:center;">
          <div style="font-size:1rem;margin-bottom:0.5rem;color:#6e7681;">
            Physiological Analysis
          </div>
          <div style="font-size:0.9rem;">
            Signal not available — {reason}
          </div>
          <div style="font-size:0.78rem;margin-top:0.5rem;color:#484f58;">
            Videos under 8 seconds, with poor face tracking, or low quality
            signals are excluded rather than fabricated.
          </div>
        </div>"""

    feats = result.rppg_features
    pulse_bpm  = feats.get('pulse_rate_bpm_proxy', 0)
    pulse_std  = feats.get('pulse_std', 0)
    spec_ent   = feats.get('spectral_entropy', 0)
    band_pwr   = feats.get('band_power', 0)
    pk_ratio   = feats.get('peak_power_ratio', 0)
    skew       = feats.get('pulse_skew', 0)
    kurtosis   = feats.get('pulse_kurtosis', 0)
    rms        = feats.get('pulse_rms', 0)

    prob_pct = _fmt_pct(result.rppg_prob)
    color = "#f85149" if result.rppg_label == "FAKE" else "#39d353"

    rows = "".join([
        f"<tr><td style='color:#8b949e;padding:0.3rem 0.5rem;'>{name}</td>"
        f"<td style='color:#e6edf3;padding:0.3rem 0.5rem;text-align:right;'>{val:.4f}</td></tr>"
        for name, val in [
            ("Pulse rate proxy (BPM)", pulse_bpm),
            ("Band power", band_pwr),
            ("Peak power ratio", pk_ratio),
            ("Spectral entropy", spec_ent),
            ("Pulse std", pulse_std),
            ("Pulse RMS", rms),
            ("Pulse skewness", skew),
            ("Pulse kurtosis", kurtosis),
        ]
    ])

    return f"""
    <div style="background:#161b22;border:1px solid #21262d;border-radius:10px;padding:1.2rem;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
        <div>
          <div style="color:#8b949e;font-size:0.78rem;text-transform:uppercase;letter-spacing:.05em;">
            Model B — rPPG Physiological
          </div>
          <div style="color:{color};font-size:1.3rem;font-weight:700;margin-top:0.2rem;">
            {result.rppg_label} &nbsp;<span style="font-size:0.9rem;font-weight:400;">({prob_pct})</span>
          </div>
        </div>
        <div style="color:#6e7681;font-size:0.78rem;text-align:right;">
          Threshold: 0.70<br>ExtraTrees classifier
        </div>
      </div>
      <table style="width:100%;border-collapse:collapse;font-size:0.82rem;">
        {rows}
      </table>
      <div style="margin-top:0.75rem;padding:0.5rem;background:#0d1117;border-radius:6px;">
        <span style="color:#6e7681;font-size:0.75rem;">
          rPPG features are proxy signals derived from facial colour variation.
          Signal quality depends on video duration, frame rate and lighting.
        </span>
      </div>
    </div>"""


def build_model_a_metrics_html() -> str:
    m = _a_metrics
    if not m:
        return "<p style='color:#8b949e;'>Model A metrics not found.</p>"

    cards_html = "".join([
        _metric_card("Accuracy",         _fmt_pct(m.get("accuracy", 0)),          "#58a6ff"),
        _metric_card("Balanced Acc.",     _fmt_pct(m.get("balanced_accuracy", 0)), "#58a6ff"),
        _metric_card("F1 (FAKE)",         _fmt_pct(m.get("f1_fake", 0)),           "#39d353"),
        _metric_card("ROC-AUC",           f"{m.get('roc_auc', 0):.3f}",            "#d29922"),
        _metric_card("Precision (FAKE)",  _fmt_pct(m.get("precision_fake", 0)),    "#39d353"),
        _metric_card("Recall (FAKE)",     _fmt_pct(m.get("recall_fake", 0)),       "#39d353"),
        _metric_card("Test videos",       str(m.get("test_videos", "?")),          "#8b949e"),
        _metric_card("Best epoch",        str(m.get("best_epoch", "?")),           "#8b949e"),
    ])

    return f"""
    <div style="margin-bottom:1rem;">
      <div style="color:#58a6ff;font-size:0.82rem;font-weight:600;margin-bottom:0.75rem;
                  text-transform:uppercase;letter-spacing:.05em;">
        Model A — ResNeXt50 + LSTM (Visual)
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:0.6rem;">{cards_html}</div>
      <div style="margin-top:0.6rem;color:#6e7681;font-size:0.75rem;">
        Threshold: {m.get('threshold_selected_on_validation', '?')} (selected on validation set)
        | Test size: {m.get('test_videos', '?')} videos
        | Architecture: ResNeXt50_32x4d + LSTM(hidden=256) + MLP(256,128,1)
      </div>
    </div>"""


def build_model_b_metrics_html() -> str:
    m = _b_metrics
    if not m:
        return "<p style='color:#8b949e;'>Model B metrics not found.</p>"

    cards_html = "".join([
        _metric_card("Accuracy",        _fmt_pct(m.get("accuracy", 0)),          "#58a6ff"),
        _metric_card("Balanced Acc.",   _fmt_pct(m.get("balanced_accuracy", 0)), "#58a6ff"),
        _metric_card("F1 (FAKE)",       _fmt_pct(m.get("f1_fake", 0)),           "#39d353"),
        _metric_card("ROC-AUC",         f"{m.get('roc_auc', 0):.3f}",            "#d29922"),
        _metric_card("Precision (FAKE)", _fmt_pct(m.get("precision_fake", 0)),   "#39d353"),
        _metric_card("Recall (FAKE)",   _fmt_pct(m.get("recall_fake", 0)),       "#39d353"),
        _metric_card("Test videos",     str(m.get("n_test", "?")),               "#8b949e"),
        _metric_card("Excl. (quality)", str(m.get("excluded_by_quality", {}).get("test", "?")), "#8b949e"),
    ])

    return f"""
    <div style="margin-bottom:1rem;">
      <div style="color:#39d353;font-size:0.82rem;font-weight:600;margin-bottom:0.75rem;
                  text-transform:uppercase;letter-spacing:.05em;">
        Model B — CHROM rPPG Physiological Classifier
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:0.6rem;">{cards_html}</div>
      <div style="margin-top:0.6rem;color:#6e7681;font-size:0.75rem;">
        Threshold: {m.get('validation_threshold', '?')} (selected on validation set)
        | Classifier: {m.get('selected_classifier', '?')}
        | 8 CHROM-rPPG features
      </div>
    </div>"""


def build_model_c_metrics_html() -> str:
    m = _c_metrics
    if not m:
        return "<p style='color:#8b949e;'>Model C metrics not found.</p>"

    hybrid = m.get("hybrid_fusion", {})
    visual = m.get("visual_only", {})
    rppg   = m.get("rppg_only", {})

    def row(name, metric_key, fmt=_fmt_pct, is_auc=False):
        def fmtv(d, k):
            v = d.get(k, None)
            if v is None:
                return "—"
            if is_auc:
                return f"{v:.3f}"
            return fmt(v)
        h_val = fmtv(hybrid, metric_key)
        v_val = fmtv(visual, metric_key)
        r_val = fmtv(rppg, metric_key)
        return (
            f"<tr>"
            f"<td style='padding:0.3rem 0.5rem;color:#8b949e;'>{name}</td>"
            f"<td style='padding:0.3rem 0.5rem;color:#58a6ff;text-align:right;'>{h_val}</td>"
            f"<td style='padding:0.3rem 0.5rem;color:#39d353;text-align:right;'>{v_val}</td>"
            f"<td style='padding:0.3rem 0.5rem;color:#d29922;text-align:right;'>{r_val}</td>"
            f"</tr>"
        )

    rows = (
        row("Accuracy",           "accuracy") +
        row("Balanced Accuracy",  "balanced_accuracy") +
        row("F1 FAKE",            "f1_fake") +
        row("ROC-AUC",            "roc_auc", is_auc=True) +
        row("Precision (FAKE)",   "precision_fake") +
        row("Recall (FAKE)",      "recall_fake")
    )

    return f"""
    <div style="margin-bottom:1rem;">
      <div style="color:#d29922;font-size:0.82rem;font-weight:600;margin-bottom:0.75rem;
                  text-transform:uppercase;letter-spacing:.05em;">
        Model C — Hybrid Fusion Comparison
      </div>
      <table style="width:100%;border-collapse:collapse;font-size:0.82rem;">
        <thead>
          <tr>
            <th style="padding:0.3rem 0.5rem;color:#6e7681;text-align:left;">Metric</th>
            <th style="padding:0.3rem 0.5rem;color:#58a6ff;text-align:right;">Hybrid (264d)</th>
            <th style="padding:0.3rem 0.5rem;color:#39d353;text-align:right;">Visual (256d)</th>
            <th style="padding:0.3rem 0.5rem;color:#d29922;text-align:right;">rPPG (8d)</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
      <div style="margin-top:0.6rem;color:#6e7681;font-size:0.75rem;">
        Hybrid = 256-dim visual embedding + 8 rPPG features.
        Logistic regression with StandardScaler.
        Test set: 33 videos (common subset after rPPG quality filtering).
      </div>
    </div>"""


# ---------------------------------------------------------------------------
# Main inference handler
# ---------------------------------------------------------------------------
def run_analysis(video_file):
    """Called by Gradio when user clicks Analyse."""
    if video_file is None:
        empty_verdict = "<p style='color:#6e7681;padding:2rem;text-align:center;'>Upload a video and click Analyse.</p>"
        return empty_verdict, "", [], "", []

    try:
        pipeline = get_pipeline()
        result   = pipeline.run(video_path=video_file)

        verdict_html = build_verdict_html(result)
        status_text  = build_status_html(result)
        face_pil     = [Image.fromarray(c) for c in result.face_crops] if result.face_crops else []
        rppg_html    = build_rppg_html(result)

        # Grad-CAM (best-effort — may fail on CPU with complex computation)
        gradcam_imgs = []
        try:
            from src.explainability import try_gradcam
            from src.video_processor import extract_frames
            from src.face_detector import preprocess_frames
            frames_bgr, _ = extract_frames(video_file)
            tensor, _     = preprocess_frames(frames_bgr)
            from src.config import get_device
            device = get_device()
            tensor = tensor.to(device)
            raw_maps = try_gradcam(pipeline.model_a, tensor, device, max_maps=6)
            gradcam_imgs = [Image.fromarray(m) for m in raw_maps]
        except Exception as e:
            logger.warning(f"Grad-CAM skipped: {e}")

        return verdict_html, status_text, face_pil, rppg_html, gradcam_imgs

    except Exception as e:
        logger.exception("Analysis failed")
        err_html = f"""
        <div style="background:#3d1f1f;border:1px solid #f85149;border-radius:10px;
                    padding:1rem;color:#f85149;font-family:monospace;font-size:0.85rem;">
          Error: {str(e)}
        </div>"""
        return err_html, f"Error: {e}", [], "", []


# ---------------------------------------------------------------------------
# Architecture HTML
# ---------------------------------------------------------------------------
ARCHITECTURE_HTML = """
<div style="font-family:'Segoe UI',sans-serif;color:#e6edf3;padding:0.5rem;">

  <div style="text-align:center;padding:1.5rem 0 1rem;">
    <div style="color:#58a6ff;font-size:1.8rem;font-weight:800;letter-spacing:.02em;">
      DeepShield
    </div>
    <div style="color:#8b949e;font-size:0.95rem;margin-top:0.3rem;">
      Multimodal Deepfake Detection and Video Forensic Analysis
    </div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin:1rem 0;">
    <div style="background:#161b22;border:1px solid #21262d;border-radius:10px;
                padding:1rem;border-top:3px solid #58a6ff;">
      <div style="color:#58a6ff;font-weight:700;margin-bottom:0.5rem;">Model A — Visual</div>
      <div style="color:#8b949e;font-size:0.82rem;line-height:1.6;">
        ResNeXt50_32x4d CNN extracts 2048-dim features per frame.
        Single-layer LSTM (hidden=256) models temporal dependencies across
        16 uniformly sampled frames. MLP classifier outputs P(FAKE).
        Threshold: 0.51 (validation-tuned).
      </div>
    </div>
    <div style="background:#161b22;border:1px solid #21262d;border-radius:10px;
                padding:1rem;border-top:3px solid #39d353;">
      <div style="color:#39d353;font-weight:700;margin-bottom:0.5rem;">Model B — Physiological</div>
      <div style="color:#8b949e;font-size:0.82rem;line-height:1.6;">
        CHROM rPPG algorithm extracts photoplethysmography proxy signals from
        facial skin colour variation. Welch PSD yields 8 spectral/temporal features.
        ExtraTrees classifier with threshold 0.70.
        Videos with insufficient signal quality are excluded.
      </div>
    </div>
    <div style="background:#161b22;border:1px solid #21262d;border-radius:10px;
                padding:1rem;border-top:3px solid #d29922;">
      <div style="color:#d29922;font-weight:700;margin-bottom:0.5rem;">Model C — Hybrid Fusion</div>
      <div style="color:#8b949e;font-size:0.82rem;line-height:1.6;">
        Concatenates the 256-dim visual embedding (from Model A's LSTM) with
        the 8 rPPG features, forming a 264-dim representation.
        Logistic regression with StandardScaler.
        Falls back to visual-only (256d, threshold 0.52) when rPPG is unavailable.
      </div>
    </div>
  </div>

  <div style="background:#161b22;border:1px solid #21262d;border-radius:10px;
              padding:1rem;margin-bottom:1rem;">
    <div style="color:#e6edf3;font-weight:600;margin-bottom:0.5rem;">Dataset — FaceForensics++</div>
    <div style="color:#8b949e;font-size:0.82rem;line-height:1.6;">
      Trained on the FaceForensics++ dataset (c23 compression subset).
      Contains videos manipulated by DeepFakes, Face2Face, FaceSwap and NeuralTextures,
      alongside pristine source videos. Training split: 285 / 74 / 41 (train/val/test).
      Hybrid model common subset: 246 / 68 / 33 after rPPG quality filtering.
    </div>
  </div>

  <div style="background:#3d2a00;border:1px solid #d29922;border-radius:10px;
              padding:1rem;margin-bottom:1rem;">
    <div style="color:#d29922;font-weight:600;margin-bottom:0.5rem;">Known Limitations</div>
    <ul style="color:#8b949e;font-size:0.82rem;line-height:1.9;margin:0;padding-left:1.2rem;">
      <li>All models show FAKE-class bias due to class imbalance in the training data.</li>
      <li>Model A test accuracy is 71% on 41 videos — small test set, low statistical reliability.</li>
      <li>rPPG signal is a physiological proxy, not a measured heartbeat waveform.</li>
      <li>Model may not generalise to manipulation methods outside FaceForensics++.</li>
      <li>Performance degrades on highly compressed, low-resolution or short videos.</li>
      <li>This is a research prototype for academic purposes. Not production-ready.</li>
    </ul>
  </div>

  <div style="background:#161b22;border:1px solid #21262d;border-radius:10px;padding:1rem;">
    <div style="color:#e6edf3;font-weight:600;margin-bottom:0.75rem;">Team</div>
    <div style="display:flex;flex-wrap:wrap;gap:0.75rem;">
      <div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:0.6rem 0.9rem;">
        <div style="color:#58a6ff;font-size:0.82rem;font-weight:600;">Mohammed Owaiz Sherrif</div>
        <div style="color:#6e7681;font-size:0.75rem;">2582425</div>
      </div>
      <div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:0.6rem 0.9rem;">
        <div style="color:#58a6ff;font-size:0.82rem;font-weight:600;">Darshan Gowda</div>
        <div style="color:#6e7681;font-size:0.75rem;">2582439</div>
      </div>
      <div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:0.6rem 0.9rem;">
        <div style="color:#58a6ff;font-size:0.82rem;font-weight:600;">Pitaliya Kanika Shantilal</div>
        <div style="color:#6e7681;font-size:0.75rem;">2582441</div>
      </div>
    </div>
    <div style="color:#6e7681;font-size:0.75rem;margin-top:0.75rem;">
      CHRIST University &nbsp;|&nbsp; MDA507-5 — Neural Networks and Deep Learning
    </div>
  </div>
</div>
"""


# ---------------------------------------------------------------------------
# Build Gradio interface
# ---------------------------------------------------------------------------
def build_interface():
    with gr.Blocks(title="DeepShield — Deepfake Detection") as demo:

        # Header
        gr.HTML("""
        <div style="background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);
                    padding:1rem 1.5rem;display:flex;align-items:center;gap:0.75rem;
                    border-bottom:1px solid #21262d;margin-bottom:0.5rem;">
          <div>
            <div style="color:#58a6ff;font-size:1.1rem;font-weight:700;
                        letter-spacing:.02em;">DeepShield</div>
            <div style="color:#6e7681;font-size:0.75rem;">
              Multimodal Deepfake Detection and Video Forensic Analysis
              &nbsp;&middot;&nbsp; CHRIST University MDA507-5
            </div>
          </div>
        </div>
        """)

        with gr.Tabs():

            # ================================================================
            # TAB 1: Video Analysis
            # ================================================================
            with gr.Tab("Video Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(
                            label="Upload Video",
                            sources=["upload"],
                        )
                        analyse_btn = gr.Button(
                            "Analyse Video",
                            variant="primary",
                            size="lg",
                        )
                        status_text = gr.Textbox(
                            label="Analysis Summary",
                            interactive=False,
                            lines=2,
                        )

                    with gr.Column(scale=1):
                        verdict_html = gr.HTML(
                            value="<p style='color:#6e7681;padding:2rem;text-align:center;'>"
                                  "Upload a video and click Analyse.</p>"
                        )

                # State stores
                face_crops_state  = gr.State([])
                rppg_html_state   = gr.State("")
                gradcam_state     = gr.State([])

                analyse_btn.click(
                    fn=run_analysis,
                    inputs=[video_input],
                    outputs=[verdict_html, status_text, face_crops_state,
                             rppg_html_state, gradcam_state],
                )

                gr.Markdown(
                    "> **Supported formats:** MP4, AVI, MOV, MKV, WebM, M4V &nbsp;·&nbsp; "
                    "**Max size:** 500 MB &nbsp;·&nbsp; "
                    "**Min duration for rPPG:** 8 seconds",
                    elem_classes=["card"],
                )

            # ================================================================
            # TAB 2: Visual Forensics
            # ================================================================
            with gr.Tab("Visual Forensics"):
                gr.Markdown("""
### Face Crops and Grad-CAM Heatmaps

Face crops below show the 224×224 inputs fed to the ResNeXt50 encoder after Haar Cascade
face detection and ImageNet normalization.

Grad-CAM heatmaps (if shown) highlight spatial regions the CNN focuses on.
The prediction is **video-level** — the LSTM processes all 16 frames as a sequence.
No per-frame deepfake probability is available from this architecture.
                """)

                refresh_btn = gr.Button("Refresh from last analysis", size="sm")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Face Crops (model inputs)")
                        face_gallery = gr.Gallery(
                            label="Extracted Face Crops",
                            columns=4, rows=2,
                            object_fit="contain",
                            height=280,
                        )
                    with gr.Column():
                        gr.Markdown("#### Grad-CAM — CNN Spatial Attention")
                        gradcam_gallery = gr.Gallery(
                            label="Grad-CAM Heatmaps",
                            columns=4, rows=2,
                            object_fit="contain",
                            height=280,
                        )

                def refresh_forensics(faces, gradcams):
                    return faces or [], gradcams or []

                refresh_btn.click(
                    fn=refresh_forensics,
                    inputs=[face_crops_state, gradcam_state],
                    outputs=[face_gallery, gradcam_gallery],
                )

                # Also update automatically after analysis
                analyse_btn.click(
                    fn=refresh_forensics,
                    inputs=[face_crops_state, gradcam_state],
                    outputs=[face_gallery, gradcam_gallery],
                )

            # ================================================================
            # TAB 3: Physiological Analysis
            # ================================================================
            with gr.Tab("Physiological Analysis"):
                gr.Markdown("""
### CHROM rPPG Signal Analysis — Model B

The physiological branch estimates a proxy pulse signal from facial skin colour variation
(Chrominance-based rPPG / CHROM). Eight spectral and temporal features are extracted and
passed to an ExtraTrees classifier trained independently from the visual model.

Videos under 8 seconds, with poor face tracking coverage (<80%) or low-quality signals
are excluded rather than assigned fabricated features.
                """)

                rppg_refresh_btn = gr.Button("Refresh from last analysis", size="sm")
                rppg_output_html = gr.HTML(
                    value="<p style='color:#6e7681;padding:1rem;'>Run an analysis to see physiological results.</p>"
                )

                def refresh_rppg(rppg_html):
                    return rppg_html or "<p style='color:#6e7681;padding:1rem;'>No analysis yet.</p>"

                rppg_refresh_btn.click(
                    fn=refresh_rppg,
                    inputs=[rppg_html_state],
                    outputs=[rppg_output_html],
                )

                analyse_btn.click(
                    fn=refresh_rppg,
                    inputs=[rppg_html_state],
                    outputs=[rppg_output_html],
                )

            # ================================================================
            # TAB 4: Model Performance
            # ================================================================
            with gr.Tab("Model Performance"):

                # ── Model A ──────────────────────────────────────────────────
                gr.HTML(build_model_a_metrics_html())
                with gr.Row():
                    if _a_cm:
                        gr.Image(value=_a_cm,   label="Model A — Confusion Matrix",  interactive=False, height=350)
                    if _a_roc:
                        gr.Image(value=_a_roc,  label="Model A — ROC Curve",         interactive=False, height=350)
                if _a_hist:
                    gr.Image(value=_a_hist, label="Model A — Training History", interactive=False, height=300)
                if _a_report:
                    gr.Code(value=_a_report, language=None, label="Model A — Classification Report", interactive=False)

                gr.HTML("<hr style='border-color:#21262d;margin:1.5rem 0;'>")

                # ── Model B ──────────────────────────────────────────────────
                gr.HTML(build_model_b_metrics_html())
                with gr.Row():
                    if _b_cm:
                        gr.Image(value=_b_cm,  label="Model B — Confusion Matrix",   interactive=False, height=350)
                    if _b_roc:
                        gr.Image(value=_b_roc, label="Model B — ROC Curve",          interactive=False, height=350)
                if _b_fi:
                    gr.Image(value=_b_fi,  label="Model B — Feature Importance",  interactive=False, height=350)
                if _b_report:
                    gr.Code(value=_b_report, language=None, label="Model B — Classification Report", interactive=False)

                gr.HTML("<hr style='border-color:#21262d;margin:1.5rem 0;'>")

                # ── Model C ──────────────────────────────────────────────────
                gr.HTML(build_model_c_metrics_html())
                if _c_comp:
                    gr.Image(value=_c_comp,     label="Model Comparison",             interactive=False, height=400)
                with gr.Row():
                    if _c_hybrid_cm:
                        gr.Image(value=_c_hybrid_cm,  label="Hybrid Fusion CM",   interactive=False, height=350)
                    if _c_visual_cm:
                        gr.Image(value=_c_visual_cm,  label="Visual Only CM",     interactive=False, height=350)
                    if _c_rppg_cm:
                        gr.Image(value=_c_rppg_cm,    label="rPPG Only CM",       interactive=False, height=350)

            # ================================================================
            # TAB 5: Architecture & About
            # ================================================================
            with gr.Tab("Architecture"):
                gr.HTML(ARCHITECTURE_HTML)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepShield Deepfake Detection")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port",  type=int, default=7860)
    parser.add_argument("--host",  type=str, default="127.0.0.1")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("DeepShield — Starting up")
    logger.info("=" * 60)

    # Pre-load models
    try:
        get_pipeline()
        logger.info("All models loaded successfully at startup.")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        logger.error("Application will start but analysis will fail.")

    demo = build_interface()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=False,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.gray,
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=CSS,
    )
