"""
src/ui_components.py
====================
Polished dark-themed UI components, CSS, HTML cards, and real signal plots
for the DeepShield Gradio interface.
"""

import io
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Global CSS — Full-width dark forensic AI theme
# ---------------------------------------------------------------------------
CSS = """
/* ── Full-width dark navy / charcoal base ── */
html, body, .gradio-container {
    background-color: #070b14 !important;
    color: #f8fafc !important;
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    padding: 0.75rem 2rem 2rem !important;
    margin: 0 !important;
}

.main, .wrap, .contain {
    max-width: 100% !important;
}

/* ── Tabs ── */
.tab-nav {
    border-bottom: 1px solid #1e293b !important;
    gap: 0.5rem !important;
    margin-bottom: 1.25rem !important;
    padding-bottom: 0.25rem !important;
}

.tab-nav button {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    padding: 0.65rem 1.35rem !important;
    border-radius: 8px 8px 0 0 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    transition: all 0.2s ease !important;
}

.tab-nav button:hover {
    color: #f8fafc !important;
    background: rgba(56, 189, 248, 0.08) !important;
}

.tab-nav button.selected {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
    background: rgba(56, 189, 248, 0.12) !important;
}

/* ── Panels, blocks & forms ── */
.block, .form, .panel {
    background: #0d1320 !important;
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35) !important;
}

label, .label-wrap, span[data-testid="block-info"] {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

/* ── Buttons ── */
button.primary {
    background: linear-gradient(135deg, #0284c7 0%, #06b6d4 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 1.4rem !important;
    box-shadow: 0 4px 14px rgba(6, 182, 212, 0.28) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}

button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(6, 182, 212, 0.42) !important;
}

button.secondary {
    background: #162033 !important;
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    padding: 0.7rem 1.4rem !important;
}

button.secondary:hover {
    background: #1e293b !important;
    border-color: #38bdf8 !important;
    color: #f8fafc !important;
}

/* ── Video & Gallery containers ── */
.upload-container, [data-testid="video"] {
    background: #0a101d !important;
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
}

.gallery-item {
    background: #0a101d !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
}

/* ── Markdown & prose contrast ── */
.prose, .prose p, .prose li, .prose h3, .prose h4 {
    color: #e2e8f0 !important;
}

.prose h3, .prose h4 {
    color: #f8fafc !important;
    font-weight: 700 !important;
}

/* ── Hide unnecessary Gradio footer ── */
footer {
    display: none !important;
}
"""


# ---------------------------------------------------------------------------
# Header & Empty States
# ---------------------------------------------------------------------------
HEADER_HTML = """
<div style="background:linear-gradient(135deg,#0b1220 0%,#111c33 60%,#0b1528 100%);
            border:1px solid #1e293b;border-radius:14px;
            padding:1.15rem 1.6rem;margin-bottom:1rem;
            display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem;
            box-shadow:0 8px 28px rgba(0,0,0,0.4);">
  <div style="display:flex;align-items:center;gap:1rem;">
    <div style="width:42px;height:42px;border-radius:10px;
                background:linear-gradient(135deg,#0284c7,#22d3ee);
                display:flex;align-items:center;justify-content:center;
                font-weight:800;font-size:1.15rem;color:#070b14;
                box-shadow:0 0 18px rgba(34,211,238,0.35);">DS</div>
    <div>
      <div style="color:#f8fafc;font-size:1.45rem;font-weight:800;letter-spacing:-0.01em;line-height:1.15;">
        DeepShield
      </div>
      <div style="color:#38bdf8;font-size:0.9rem;font-weight:500;margin-top:0.15rem;">
        Multimodal Deepfake Detection and Video Forensic Analysis
      </div>
    </div>
  </div>
  <div style="display:flex;gap:0.6rem;flex-wrap:wrap;">
    <span style="background:#0f172a;border:1px solid #1e293b;color:#cbd5e1;
                 padding:0.35rem 0.75rem;border-radius:999px;font-size:0.78rem;font-weight:600;">
      Visual: ResNeXt50 + LSTM
    </span>
    <span style="background:#0f172a;border:1px solid #1e293b;color:#cbd5e1;
                 padding:0.35rem 0.75rem;border-radius:999px;font-size:0.78rem;font-weight:600;">
      Physiological: CHROM-rPPG
    </span>
    <span style="background:#0f172a;border:1px solid #0284c7;color:#38bdf8;
                 padding:0.35rem 0.75rem;border-radius:999px;font-size:0.78rem;font-weight:600;">
      264-D Hybrid Fusion
    </span>
  </div>
</div>
"""

EMPTY_VERDICT_HTML = """
<div style="background:linear-gradient(180deg,#0d1320 0%,#0a101d 100%);
            border:1px solid #1e293b;border-radius:14px;padding:2.4rem 1.8rem;
            text-align:center;box-shadow:0 8px 28px rgba(0,0,0,0.35);">
  <div style="display:inline-block;background:rgba(56,189,248,0.12);border:1px solid rgba(56,189,248,0.35);
              color:#38bdf8;font-size:0.78rem;font-weight:700;letter-spacing:0.08em;
              text-transform:uppercase;padding:0.35rem 0.9rem;border-radius:999px;margin-bottom:1rem;">
    Ready for Forensic Analysis
  </div>
  <div style="color:#f8fafc;font-size:1.35rem;font-weight:700;margin-bottom:0.5rem;">
    Upload a Video to Begin
  </div>
  <div style="color:#cbd5e1;font-size:0.92rem;max-width:460px;margin:0 auto 1.5rem;line-height:1.55;">
    DeepShield evaluates spatial-temporal facial artifacts alongside physiological pulse signals
    to produce a calibrated authenticity verdict.
  </div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.75rem;text-align:left;">
    <div style="background:#111827;border:1px solid #1e293b;border-radius:10px;padding:0.85rem;">
      <div style="color:#38bdf8;font-size:0.78rem;font-weight:700;text-transform:uppercase;">1. Visual</div>
      <div style="color:#e2e8f0;font-size:0.82rem;margin-top:0.25rem;">16-frame face sequence via ResNeXt50 + LSTM</div>
    </div>
    <div style="background:#111827;border:1px solid #1e293b;border-radius:10px;padding:0.85rem;">
      <div style="color:#2dd4bf;font-size:0.78rem;font-weight:700;text-transform:uppercase;">2. rPPG</div>
      <div style="color:#e2e8f0;font-size:0.82rem;margin-top:0.25rem;">CHROM pulse extraction &amp; Welch PSD</div>
    </div>
    <div style="background:#111827;border:1px solid #1e293b;border-radius:10px;padding:0.85rem;">
      <div style="color:#f8fafc;font-size:0.78rem;font-weight:700;text-transform:uppercase;">3. Fusion</div>
      <div style="color:#e2e8f0;font-size:0.82rem;margin-top:0.25rem;">Hybrid fusion or automatic visual fallback</div>
    </div>
  </div>
</div>
"""

EMPTY_STATUS_HTML = """
<div style="background:#0d1320;border:1px solid #1e293b;border-radius:10px;
            padding:0.75rem 1rem;color:#cbd5e1;font-size:0.86rem;">
  <strong style="color:#38bdf8;">Status:</strong> Waiting for video upload.
</div>
"""

EMPTY_RPPG_HTML = """
<div style="background:#0d1320;border:1px solid #1e293b;border-radius:12px;
            padding:1.8rem;text-align:center;">
  <div style="color:#f8fafc;font-size:1.05rem;font-weight:700;margin-bottom:0.35rem;">
    Physiological Signal Analysis
  </div>
  <div style="color:#cbd5e1;font-size:0.9rem;">
    Run an analysis on the <strong>Analyze Video</strong> tab to view extracted CHROM-rPPG features and waveforms.
  </div>
</div>
"""


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


# ---------------------------------------------------------------------------
# Hero Result Card
# ---------------------------------------------------------------------------
def build_verdict_html(result) -> str:
    """Build high-impact hero result card from InferenceResult."""
    if result.route == "error" or result.label == "INCONCLUSIVE":
        accent = "#fbbf24"
        bg_grad = "linear-gradient(135deg,#2a1f06 0%,#16120a 100%)"
        border = "#f59e0b"
        verdict_text = "INCONCLUSIVE"
    elif result.is_fake:
        accent = "#f87171"
        bg_grad = "linear-gradient(135deg,#331218 0%,#160b10 100%)"
        border = "#ef4444"
        verdict_text = "FAKE"
    else:
        accent = "#2dd4bf"
        bg_grad = "linear-gradient(135deg,#0b2926 0%,#08171a 100%)"
        border = "#10b981"
        verdict_text = "REAL"

    route_styles = {
        "hybrid_fusion": ("Hybrid Fusion", "#38bdf8", "rgba(56,189,248,0.14)", "#0284c7"),
        "visual_only":   ("Visual Only Fallback", "#fbbf24", "rgba(245,158,11,0.14)", "#d97706"),
        "error":         ("Inconclusive", "#fbbf24", "rgba(245,158,11,0.14)", "#d97706"),
    }
    route_label, r_color, r_bg, r_border = route_styles.get(
        result.route, ("Visual Only Fallback", "#fbbf24", "rgba(245,158,11,0.14)", "#d97706")
    )

    pct = _fmt_pct(result.confidence)
    bar_width = max(2, min(100, int(round(result.confidence * 100))))

    hybrid_val = _fmt_pct(result.confidence) if result.route == "hybrid_fusion" else "Fallback"
    visual_val = _fmt_pct(result.visual_prob)
    rppg_val = _fmt_pct(result.rppg_prob) if result.rppg_available and result.rppg_prob is not None else "Unavailable"

    warn_html = ""
    if result.warning:
        warn_html = f"""
        <div style="background:rgba(245,158,11,0.12);border:1px solid #d97706;border-radius:10px;
                    padding:0.8rem 1rem;margin-top:0.9rem;text-align:left;">
          <div style="color:#fbbf24;font-size:0.84rem;font-weight:600;line-height:1.45;">
            Notice: {result.warning}
          </div>
        </div>"""

    return f"""
    <div style="background:{bg_grad};border:2px solid {border};border-radius:14px;
                padding:1.6rem 1.5rem;box-shadow:0 10px 30px rgba(0,0,0,0.45);">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.75rem;">
        <span style="color:#cbd5e1;font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;">
          Forensic Verdict
        </span>
        <span style="background:{r_bg};border:1px solid {r_border};color:{r_color};
                     padding:0.3rem 0.8rem;border-radius:999px;font-size:0.78rem;font-weight:700;">
          Route: {route_label}
        </span>
      </div>

      <div style="display:flex;align-items:baseline;justify-content:space-between;flex-wrap:wrap;gap:1rem;margin-top:0.85rem;">
        <div style="font-size:2.9rem;font-weight:900;color:{accent};letter-spacing:0.02em;line-height:1;">
          {verdict_text}
        </div>
        <div style="text-align:right;">
          <div style="color:#cbd5e1;font-size:0.78rem;font-weight:600;text-transform:uppercase;">
            Fake Probability Score
          </div>
          <div style="color:#f8fafc;font-size:1.75rem;font-weight:800;line-height:1.15;">
            {pct}
          </div>
        </div>
      </div>

      <div style="background:#070b14;border:1px solid #1e293b;border-radius:999px;height:10px;
                  overflow:hidden;margin:1rem 0 1.2rem;">
        <div style="background:{accent};width:{bar_width}%;height:100%;border-radius:999px;"></div>
      </div>

      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.75rem;">
        <div style="background:rgba(13,19,32,0.85);border:1px solid #1e293b;border-radius:10px;padding:0.8rem;text-align:center;">
          <div style="color:#cbd5e1;font-size:0.74rem;font-weight:600;text-transform:uppercase;">Hybrid Fusion</div>
          <div style="color:#38bdf8;font-size:1.15rem;font-weight:800;margin-top:0.2rem;">{hybrid_val}</div>
        </div>
        <div style="background:rgba(13,19,32,0.85);border:1px solid #1e293b;border-radius:10px;padding:0.8rem;text-align:center;">
          <div style="color:#cbd5e1;font-size:0.74rem;font-weight:600;text-transform:uppercase;">Visual (Model A)</div>
          <div style="color:#f8fafc;font-size:1.15rem;font-weight:800;margin-top:0.2rem;">{visual_val}</div>
        </div>
        <div style="background:rgba(13,19,32,0.85);border:1px solid #1e293b;border-radius:10px;padding:0.8rem;text-align:center;">
          <div style="color:#cbd5e1;font-size:0.74rem;font-weight:600;text-transform:uppercase;">rPPG (Model B)</div>
          <div style="color:#2dd4bf;font-size:1.15rem;font-weight:800;margin-top:0.2rem;">{rppg_val}</div>
        </div>
      </div>

      {warn_html}

      <div style="margin-top:0.95rem;padding-top:0.75rem;border-top:1px solid rgba(148,163,184,0.2);
                  color:#cbd5e1;font-size:0.78rem;line-height:1.45;">
        Confidence is a calibrated model probability, not cryptographic proof of authenticity.
      </div>
    </div>
    """


def build_status_html(result) -> str:
    duration = result.video_meta.get("duration_sec", 0)
    fps = result.video_meta.get("fps", 0)
    res = f"{result.video_meta.get('width', '?')}x{result.video_meta.get('height', '?')}"
    items = [
        ("Frames", f"{result.total_frames}"),
        ("Faces", f"{result.faces_detected}/{result.total_frames}"),
        ("Duration", f"{duration:.1f}s"),
        ("FPS", f"{fps:.1f}"),
        ("Resolution", res),
        ("Runtime", f"{result.inference_time_sec:.1f}s"),
    ]
    pills = "".join(
        f"<div style='background:#111827;border:1px solid #1e293b;border-radius:8px;padding:0.45rem 0.7rem;'>"
        f"<div style='color:#94a3b8;font-size:0.7rem;font-weight:600;text-transform:uppercase;'>{k}</div>"
        f"<div style='color:#f8fafc;font-size:0.88rem;font-weight:700;'>{v}</div>"
        f"</div>"
        for k, v in items
    )
    return f"<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:0.5rem;'>{pills}</div>"


# ---------------------------------------------------------------------------
# Physiological Analysis HTML + Real Signal Plot
# ---------------------------------------------------------------------------
def build_rppg_html(result) -> str:
    if not result.rppg_available:
        reason = result.rppg_status.replace("_", " ").capitalize()
        return f"""
        <div style="background:linear-gradient(135deg,#271c06 0%,#14110b 100%);
                    border:1px solid #d97706;border-radius:12px;padding:1.4rem 1.5rem;">
          <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.5rem;">
            <div style="color:#fbbf24;font-size:1.05rem;font-weight:800;">
              Physiological Signal Unavailable
            </div>
            <span style="background:rgba(245,158,11,0.16);border:1px solid #d97706;color:#fbbf24;
                         padding:0.25rem 0.7rem;border-radius:999px;font-size:0.76rem;font-weight:700;">
              Visual-Only Fallback Active
            </span>
          </div>
          <div style="color:#f8fafc;font-size:0.92rem;margin-top:0.65rem;">
            <strong>Quality Gate Reason:</strong> {reason}
          </div>
          <div style="color:#cbd5e1;font-size:0.84rem;margin-top:0.45rem;line-height:1.5;">
            CHROM-rPPG requires at least 8.0 seconds of continuous video with &ge;80% frontal face tracking.
            DeepShield excludes low-quality signals rather than fabricating physiological metrics.
          </div>
        </div>"""

    feats = result.rppg_features or {}
    prob_pct = _fmt_pct(result.rppg_prob)
    verdict_color = "#f87171" if result.rppg_label == "FAKE" else "#2dd4bf"

    cards = "".join(
        f"<div style='background:#111827;border:1px solid #1e293b;border-radius:10px;padding:0.8rem 0.9rem;'>"
        f"<div style='color:#94a3b8;font-size:0.74rem;font-weight:600;text-transform:uppercase;'>{label}</div>"
        f"<div style='color:#f8fafc;font-size:1.1rem;font-weight:800;margin-top:0.2rem;'>{val:.4f}</div>"
        f"</div>"
        for label, val in [
            ("Pulse Rate Proxy (BPM)", feats.get("pulse_rate_bpm_proxy", 0.0)),
            ("Band Power (0.7-3.0 Hz)", feats.get("band_power", 0.0)),
            ("Peak Power Ratio", feats.get("peak_power_ratio", 0.0)),
            ("Spectral Entropy", feats.get("spectral_entropy", 0.0)),
            ("Pulse Std Dev", feats.get("pulse_std", 0.0)),
            ("Pulse RMS", feats.get("pulse_rms", 0.0)),
            ("Pulse Skewness", feats.get("pulse_skew", 0.0)),
            ("Pulse Kurtosis", feats.get("pulse_kurtosis", 0.0)),
        ]
    )

    return f"""
    <div style="background:#0d1320;border:1px solid #1e293b;border-radius:12px;padding:1.35rem 1.4rem;">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.75rem;margin-bottom:1rem;">
        <div>
          <div style="color:#38bdf8;font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;">
            Model B — CHROM-rPPG Classifier
          </div>
          <div style="color:{verdict_color};font-size:1.45rem;font-weight:800;margin-top:0.15rem;">
            {result.rppg_label} <span style="color:#e2e8f0;font-size:0.95rem;font-weight:600;">({prob_pct} FAKE score)</span>
          </div>
        </div>
        <span style="background:rgba(45,212,191,0.14);border:1px solid #10b981;color:#2dd4bf;
                     padding:0.3rem 0.8rem;border-radius:999px;font-size:0.78rem;font-weight:700;">
          Signal Quality: Passed
        </span>
      </div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.65rem;">
        {cards}
      </div>
      <div style="margin-top:0.9rem;color:#cbd5e1;font-size:0.8rem;">
        Extracted from central facial skin ROI using windowed CHROM chrominance and Welch power spectral density.
      </div>
    </div>"""


def build_rppg_plot(result):
    """Render real CHROM pulse waveform & Welch PSD as a PIL Image, or None if unavailable."""
    if not result.rppg_available or not result.rppg_signal:
        return None

    sig = result.rppg_signal
    pulse = sig["pulse"]
    fs = sig["sample_fps"]
    freqs = sig["freqs"]
    power = sig["power"]
    peak_hz = sig["peak_hz"]

    t = np.arange(len(pulse)) / fs
    band = (freqs >= 0.7) & (freqs <= 3.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), dpi=130)
    fig.patch.set_facecolor("#0d1320")

    for ax in axes:
        ax.set_facecolor("#070b14")
        ax.tick_params(colors="#cbd5e1", labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_color("#1e293b")
        ax.grid(color="#1e293b", linestyle="--", linewidth=0.6, alpha=0.8)

    # Left: Waveform
    axes[0].plot(t, pulse, color="#22d3ee", linewidth=1.2)
    axes[0].set_title("Extracted CHROM Pulse Waveform", color="#f8fafc", fontsize=10.5, fontweight="bold", pad=8)
    axes[0].set_xlabel("Time (seconds)", color="#cbd5e1", fontsize=9)
    axes[0].set_ylabel("Normalized Amplitude", color="#cbd5e1", fontsize=9)

    # Right: Welch PSD
    axes[1].plot(freqs[band] * 60.0, power[band], color="#38bdf8", linewidth=1.6)
    axes[1].fill_between(freqs[band] * 60.0, power[band], color="#38bdf8", alpha=0.18)
    axes[1].axvline(peak_hz * 60.0, color="#2dd4bf", linestyle="--", linewidth=1.4,
                    label=f"Peak: {peak_hz * 60.0:.1f} BPM")
    axes[1].set_title("Welch Power Spectral Density (42–180 BPM)", color="#f8fafc", fontsize=10.5, fontweight="bold", pad=8)
    axes[1].set_xlabel("Frequency (BPM)", color="#cbd5e1", fontsize=9)
    axes[1].set_ylabel("Power", color="#cbd5e1", fontsize=9)
    axes[1].legend(facecolor="#0d1320", edgecolor="#1e293b", labelcolor="#f8fafc", fontsize=8.5)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ---------------------------------------------------------------------------
# Architecture / About HTML (Clean, minimal, high contrast)
# ---------------------------------------------------------------------------
ARCHITECTURE_HTML = """
<div style="color:#f8fafc;padding:0.25rem 0;">

  <div style="background:#0d1320;border:1px solid #1e293b;border-radius:12px;padding:1.25rem 1.4rem;margin-bottom:1rem;">
    <div style="color:#38bdf8;font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.65rem;">
      Multimodal Inference Pipeline
    </div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.75rem;">
      <div style="background:#111827;border:1px solid #1e293b;border-radius:10px;padding:0.85rem;">
        <div style="color:#38bdf8;font-weight:700;font-size:0.86rem;">1. Frame Sampling</div>
        <div style="color:#cbd5e1;font-size:0.82rem;margin-top:0.25rem;">Uniformly samples 16 frames and validates video integrity.</div>
      </div>
      <div style="background:#111827;border:1px solid #1e293b;border-radius:10px;padding:0.85rem;">
        <div style="color:#38bdf8;font-weight:700;font-size:0.86rem;">2. Visual Branch</div>
        <div style="color:#cbd5e1;font-size:0.82rem;margin-top:0.25rem;">Haar Cascade crops faces (224x224) for ResNeXt50 + LSTM.</div>
      </div>
      <div style="background:#111827;border:1px solid #1e293b;border-radius:10px;padding:0.85rem;">
        <div style="color:#2dd4bf;font-weight:700;font-size:0.86rem;">3. rPPG Branch</div>
        <div style="color:#cbd5e1;font-size:0.82rem;margin-top:0.25rem;">Extracts CHROM pulse waveform &amp; 8 Welch PSD features.</div>
      </div>
      <div style="background:#111827;border:1px solid #1e293b;border-radius:10px;padding:0.85rem;">
        <div style="color:#f8fafc;font-weight:700;font-size:0.86rem;">4. Hybrid Decision</div>
        <div style="color:#cbd5e1;font-size:0.82rem;margin-top:0.25rem;">Fuses 264-D features or routes to visual-only fallback.</div>
      </div>
    </div>
  </div>

  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1rem;">
    <div style="background:#0d1320;border:1px solid #1e293b;border-top:3px solid #38bdf8;border-radius:12px;padding:1.2rem;">
      <div style="color:#38bdf8;font-size:1rem;font-weight:800;margin-bottom:0.5rem;">Model A — Visual</div>
      <div style="color:#cbd5e1;font-size:0.86rem;line-height:1.55;">
        ResNeXt50-32x4d extracts 2048-D frame embeddings across 16 sampled facial crops.
        A single-layer LSTM (hidden=256) models temporal inconsistencies and outputs a 256-D representation.
      </div>
    </div>
    <div style="background:#0d1320;border:1px solid #1e293b;border-top:3px solid #2dd4bf;border-radius:12px;padding:1.2rem;">
      <div style="color:#2dd4bf;font-size:1rem;font-weight:800;margin-bottom:0.5rem;">Model B — rPPG</div>
      <div style="color:#cbd5e1;font-size:0.86rem;line-height:1.55;">
        Windowed CHROM extracts subtle skin-color pulse variations from the central facial ROI.
        An ExtraTrees classifier scores 8 spectral and temporal pulse features.
      </div>
    </div>
    <div style="background:#0d1320;border:1px solid #1e293b;border-top:3px solid #06b6d4;border-radius:12px;padding:1.2rem;">
      <div style="color:#06b6d4;font-size:1rem;font-weight:800;margin-bottom:0.5rem;">Model C — Hybrid Fusion</div>
      <div style="color:#cbd5e1;font-size:0.86rem;line-height:1.55;">
        Concatenates the 256-D LSTM embedding with the 8-D rPPG vector (264-D) via a calibrated
        Logistic Regression pipeline, with automatic fallback to 256-D visual-only inference.
      </div>
    </div>
  </div>

  <div style="background:rgba(245,158,11,0.08);border:1px solid #d97706;border-radius:12px;padding:1.2rem 1.4rem;">
    <div style="color:#fbbf24;font-size:0.95rem;font-weight:800;margin-bottom:0.5rem;">Known Limitations</div>
    <ul style="color:#e2e8f0;font-size:0.86rem;line-height:1.75;margin:0;padding-left:1.25rem;">
      <li>Trained on FaceForensics++ (c23); generalization to unseen manipulation methods may vary.</li>
      <li>rPPG requires &ge;8 seconds of video and &ge;80% continuous frontal face tracking.</li>
      <li>Heavy video compression, low lighting, or rapid head motion degrades physiological signal quality.</li>
      <li>Grad-CAM maps visualize spatial CNN attention per frame and do not reflect LSTM temporal reasoning.</li>
      <li>Confidence scores represent model probabilities, not definitive forensic proof.</li>
    </ul>
  </div>

</div>
"""
