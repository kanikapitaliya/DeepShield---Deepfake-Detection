"""
app.py
======
DeepShield — Multimodal Deepfake Detection and Video Forensic Analysis
Polished, full-width dark-themed Gradio interface.

Sections:
  1. Analyze Video         — upload, preview, 3-model inference, hero result card, reset
  2. Visual Forensics      — sampled frames, extracted face crops, Grad-CAM heatmaps
  3. Physiological Analysis — CHROM-rPPG waveform, Welch PSD, and 8-feature breakdown
  4. About / Architecture  — Model A, Model B, Model C pipeline & known limitations

Usage:
    py app.py
    py app.py --port 7861
    py app.py --share
"""

import argparse
import logging
import gradio as gr
from PIL import Image

from src.ui_components import (
    CSS,
    HEADER_HTML,
    EMPTY_VERDICT_HTML,
    EMPTY_STATUS_HTML,
    EMPTY_RPPG_HTML,
    ARCHITECTURE_HTML,
    build_verdict_html,
    build_status_html,
    build_rppg_html,
    build_rppg_plot,
)

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
# Inference & Reset handlers
# ---------------------------------------------------------------------------
def run_analysis(video_file):
    """Run full multimodal analysis and update all UI sections."""
    if video_file is None:
        return (
            EMPTY_VERDICT_HTML,
            EMPTY_STATUS_HTML,
            [],
            [],
            [],
            EMPTY_RPPG_HTML,
            None,
        )

    try:
        pipeline = get_pipeline()
        result = pipeline.run(video_path=video_file)

        verdict_html = build_verdict_html(result)
        status_html = build_status_html(result)
        sampled_pil = [Image.fromarray(f) for f in result.sampled_frames] if result.sampled_frames else []
        face_pil = [Image.fromarray(c) for c in result.face_crops] if result.face_crops else []
        rppg_html = build_rppg_html(result)
        rppg_plot = build_rppg_plot(result)

        gradcam_imgs = []
        try:
            from src.explainability import try_gradcam
            from src.video_processor import extract_frames
            from src.face_detector import preprocess_frames
            from src.config import get_device

            frames_bgr, _ = extract_frames(video_file)
            tensor, _ = preprocess_frames(frames_bgr)
            device = get_device()
            raw_maps = try_gradcam(pipeline.model_a, tensor.to(device), device, max_maps=8)
            gradcam_imgs = [Image.fromarray(m) for m in raw_maps]
        except Exception as e:
            logger.warning(f"Grad-CAM skipped: {e}")

        return (
            verdict_html,
            status_html,
            sampled_pil,
            face_pil,
            gradcam_imgs,
            rppg_html,
            rppg_plot,
        )

    except Exception as e:
        logger.exception("Analysis failed")
        err_html = f"""
        <div style="background:#2d1216;border:1px solid #ef4444;border-radius:12px;
                    padding:1.2rem;color:#f87171;font-size:0.9rem;">
          <strong>Analysis Error:</strong> {str(e)}
        </div>"""
        err_status = f"""
        <div style="background:#2d1216;border:1px solid #ef4444;border-radius:10px;
                    padding:0.75rem 1rem;color:#f87171;font-size:0.86rem;">
          <strong>Status:</strong> Failed ({str(e)})
        </div>"""
        return err_html, err_status, [], [], [], EMPTY_RPPG_HTML, None


def reset_session():
    """Clear uploaded video and all forensic outputs."""
    return (
        None,
        EMPTY_VERDICT_HTML,
        EMPTY_STATUS_HTML,
        [],
        [],
        [],
        EMPTY_RPPG_HTML,
        None,
    )


# ---------------------------------------------------------------------------
# Build Gradio interface (4 clean tabs)
# ---------------------------------------------------------------------------
def build_interface():
    with gr.Blocks(title="DeepShield — Multimodal Deepfake Detection") as demo:

        gr.HTML(HEADER_HTML)

        with gr.Tabs():

            # ================================================================
            # TAB 1: Analyze Video
            # ================================================================
            with gr.Tab("Analyze Video"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        video_input = gr.Video(
                            label="Upload Video",
                            sources=["upload"],
                            height=340,
                        )
                        with gr.Row():
                            analyse_btn = gr.Button(
                                "Analyse Video",
                                variant="primary",
                                size="lg",
                                scale=3,
                            )
                            reset_btn = gr.Button(
                                "Reset",
                                variant="secondary",
                                size="lg",
                                scale=1,
                            )
                        status_html = gr.HTML(value=EMPTY_STATUS_HTML)
                        gr.HTML(
                            "<div style='color:#cbd5e1;font-size:0.8rem;padding:0.4rem 0.2rem;'>"
                            "Supported: MP4, AVI, MOV, MKV, WebM &nbsp;|&nbsp; Max: 500 MB &nbsp;|&nbsp; "
                            "rPPG requires &ge;8s duration"
                            "</div>"
                        )

                    with gr.Column(scale=6):
                        verdict_html = gr.HTML(value=EMPTY_VERDICT_HTML)

            # ================================================================
            # TAB 2: Visual Forensics
            # ================================================================
            with gr.Tab("Visual Forensics"):
                gr.Markdown(
                    "### Sampled Frames, Face Crops & Grad-CAM Attention\n"
                    "Representative frames from the 16-frame sequence, 224x224 Haar-cropped inputs, "
                    "and CNN spatial attention overlays (video-level prediction)."
                )
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Sampled Video Frames")
                        sampled_gallery = gr.Gallery(
                            label="Sampled Frames",
                            columns=4,
                            rows=2,
                            object_fit="contain",
                            height=270,
                        )
                    with gr.Column():
                        gr.Markdown("#### Extracted Face Crops (224x224)")
                        face_gallery = gr.Gallery(
                            label="Face Crops",
                            columns=4,
                            rows=2,
                            object_fit="contain",
                            height=270,
                        )

                gr.Markdown("#### Grad-CAM — ResNeXt50 Spatial Attention")
                gradcam_gallery = gr.Gallery(
                    label="Grad-CAM Heatmaps",
                    columns=4,
                    rows=2,
                    object_fit="contain",
                    height=270,
                )

            # ================================================================
            # TAB 3: Physiological Analysis
            # ================================================================
            with gr.Tab("Physiological Analysis"):
                gr.Markdown(
                    "### CHROM-rPPG Signal & Spectral Analysis\n"
                    "Evaluates subtle facial skin color variations to extract a proxy pulse signal "
                    "and Welch power spectral density features."
                )
                rppg_output_html = gr.HTML(value=EMPTY_RPPG_HTML)
                rppg_plot_img = gr.Image(
                    label="CHROM Pulse Waveform & Welch PSD",
                    interactive=False,
                    visible=True,
                    height=320,
                )

            # ================================================================
            # TAB 4: About / Architecture
            # ================================================================
            with gr.Tab("About / Architecture"):
                gr.HTML(ARCHITECTURE_HTML)

        # --------------------------------------------------------------------
        # Event wiring (all tabs update automatically)
        # --------------------------------------------------------------------
        analyse_btn.click(
            fn=run_analysis,
            inputs=[video_input],
            outputs=[
                verdict_html,
                status_html,
                sampled_gallery,
                face_gallery,
                gradcam_gallery,
                rppg_output_html,
                rppg_plot_img,
            ],
        )

        reset_btn.click(
            fn=reset_session,
            inputs=[],
            outputs=[
                video_input,
                verdict_html,
                status_html,
                sampled_gallery,
                face_gallery,
                gradcam_gallery,
                rppg_output_html,
                rppg_plot_img,
            ],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepShield Deepfake Detection")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    logger.info("DeepShield — Starting up")
    try:
        get_pipeline()
        logger.info("All models loaded successfully at startup.")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")

    demo = build_interface()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=False,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.cyan,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=CSS,
    )
