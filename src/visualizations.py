"""
src/visualizations.py
=====================
Load evaluation artifacts from disk and format results for display.

All metrics are read from the actual saved artifacts — nothing is fabricated.
If an artifact file is missing, a placeholder message is shown.
"""

import json
import os
import logging
import numpy as np
from PIL import Image
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def _artifact_path(filename: str) -> str:
    return os.path.join(ARTIFACTS_DIR, filename)


def load_test_metrics() -> dict:
    """
    Load test_metrics.json. Returns the dict or a dict with an 'error' key.
    """
    path = _artifact_path("test_metrics.json")
    if not os.path.isfile(path):
        return {"error": "test_metrics.json not found in artifacts/"}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


def load_classification_report() -> str:
    """Return contents of classification_report.txt or an error string."""
    path = _artifact_path("classification_report.txt")
    if not os.path.isfile(path):
        return "classification_report.txt not found in artifacts/"
    try:
        with open(path) as f:
            return f.read()
    except Exception as e:
        return str(e)


def load_image_artifact(filename: str) -> Optional[Image.Image]:
    """Load a PNG artifact image. Returns None if missing."""
    path = _artifact_path(filename)
    if not os.path.isfile(path):
        logger.warning(f"Artifact not found: {path}")
        return None
    try:
        return Image.open(path)
    except Exception as e:
        logger.warning(f"Could not load {filename}: {e}")
        return None


def format_metrics_html(metrics: dict) -> str:
    """
    Format test_metrics.json into an HTML metrics card for display.
    """
    if "error" in metrics:
        return f"<p style='color:red'>Could not load metrics: {metrics['error']}</p>"

    acc  = metrics.get("accuracy", "N/A")
    prec = metrics.get("precision", "N/A")
    rec  = metrics.get("recall", "N/A")
    f1   = metrics.get("f1", "N/A")
    auc  = metrics.get("roc_auc", "N/A")
    loss = metrics.get("test_loss", "N/A")
    n    = metrics.get("test_videos", "N/A")
    ep   = metrics.get("best_epoch", "N/A")
    thr  = metrics.get("threshold", 0.5)

    def fmt(v, pct=True):
        if isinstance(v, float):
            return f"{v*100:.1f}%" if pct else f"{v:.4f}"
        return str(v)

    html = f"""
    <div style="font-family: 'Segoe UI', sans-serif; padding: 1rem;">
      <h3 style="color:#3b82f6; margin-bottom:1rem;">📊 Evaluation Results</h3>
      <p style="color:#94a3b8; font-size:0.85rem; margin-bottom:1rem;">
        Evaluated on {n} test videos · Best checkpoint from epoch {ep} · Threshold: {thr}
      </p>
      <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:0.75rem; margin-bottom:1rem;">
        {_metric_card("Accuracy", fmt(acc), "#3b82f6")}
        {_metric_card("Precision", fmt(prec), "#8b5cf6")}
        {_metric_card("Recall", fmt(rec), "#10b981")}
        {_metric_card("F1 Score", fmt(f1), "#f59e0b")}
        {_metric_card("ROC-AUC", fmt(auc, pct=False), "#ef4444")}
        {_metric_card("Test Loss", fmt(loss, pct=False), "#6b7280")}
      </div>
      <div style="background:#fef3c7; border:1px solid #f59e0b; border-radius:8px; padding:0.75rem; margin-top:0.5rem;">
        <strong>⚠ Limitations:</strong> These results are from a 15-video test set (10 FAKE, 5 REAL),
        trained for 1 epoch. The model exhibits a strong FAKE bias (recall=1.0 for FAKE, 0.0 for REAL)
        and a below-chance ROC-AUC of 0.24. This model is a research prototype, not a production system.
      </div>
    </div>
    """
    return html


def _metric_card(label: str, value: str, color: str) -> str:
    return f"""
    <div style="background:#1e293b; border-radius:8px; padding:0.75rem; text-align:center;
                border-left:4px solid {color};">
      <div style="color:{color}; font-size:1.4rem; font-weight:700;">{value}</div>
      <div style="color:#94a3b8; font-size:0.75rem; margin-top:0.25rem;">{label}</div>
    </div>
    """


def format_prediction_html(label: str, confidence: float, warning: Optional[str] = None) -> str:
    """
    Render a styled prediction card.
    """
    if label == "FAKE":
        color = "#ef4444"
        bg = "#450a0a"
        emoji = "🚨"
    else:
        color = "#22c55e"
        bg = "#052e16"
        emoji = "✅"

    pct = confidence * 100
    bar_width = pct if label == "FAKE" else (100 - pct)
    bar_label = f"P(FAKE) = {pct:.1f}%"

    warning_html = ""
    if warning:
        warning_html = f"""
        <div style="background:#422006; border:1px solid #f97316; border-radius:6px;
                    padding:0.5rem 0.75rem; margin-top:0.75rem; color:#fdba74; font-size:0.82rem;">
          ⚠ {warning}
        </div>"""

    return f"""
    <div style="background:{bg}; border:2px solid {color}; border-radius:12px; padding:1.25rem;
                font-family:'Segoe UI', sans-serif; color:#f1f5f9;">
      <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.75rem;">
        <span style="font-size:2rem;">{emoji}</span>
        <div>
          <div style="color:{color}; font-size:1.75rem; font-weight:800; line-height:1;">{label}</div>
          <div style="color:#94a3b8; font-size:0.85rem;">Model Prediction</div>
        </div>
      </div>

      <div style="margin-bottom:0.5rem;">
        <div style="display:flex; justify-content:space-between; color:#94a3b8;
                    font-size:0.8rem; margin-bottom:0.25rem;">
          <span>{bar_label}</span>
          <span>Threshold: 50%</span>
        </div>
        <div style="background:#334155; border-radius:999px; height:10px; overflow:hidden;">
          <div style="background:{color}; width:{bar_width:.1f}%; height:100%;
                      border-radius:999px; transition:width 0.4s;"></div>
        </div>
      </div>

      <p style="color:#94a3b8; font-size:0.78rem; margin-top:0.75rem; margin-bottom:0;">
        This confidence score reflects the model's output probability, not a verified ground truth.
        The model has known limitations and should not be used as sole evidence.
      </p>
      {warning_html}
    </div>
    """
