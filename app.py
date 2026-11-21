"""Step 6 - Gradio demo: baseline vs. proposed model, with Grad-CAM heatmaps.

Local:   python app.py
Hosting: copy app.py, common.py, explain.py, requirements.txt and the two checkpoints in ckpt/
         to a Hugging Face Space (SDK: Gradio). The free CPU tier is enough for inference.
"""
from pathlib import Path

import gradio as gr

from common import CKPT_DIR, load_ckpt
from explain import gradcam

MODELS = {
    "baseline": ("Baseline: ResNet-50", CKPT_DIR / "resnet50_final.pt"),
    "proposed": ("Proposed: DINOv2 + LoRA (r=8)", CKPT_DIR / "dinov2_lora_r8_final.pt"),
}
_cache = {}

DISCLAIMER = (
    "**Research prototype - not a medical device and not for diagnosis.** "
    "Trained on 800 public chest X-rays (Montgomery + Shenzhen). Do not upload images of real patients "
    "unless you are authorised to do so; images are processed in memory and not stored."
)


def _get(key):
    if key not in _cache:
        _cache[key] = load_ckpt(MODELS[key][1], device="cpu")
    return _cache[key]


def run(image):
    if image is None:
        return ("Upload a chest X-ray.", None, "", None)
    out = []
    for key in ("baseline", "proposed"):
        model, tag, thr = _get(key)
        overlay, prob, positive = gradcam(model, tag, image, thr, device="cpu")
        verdict = "TB-suspicious (screen-positive)" if positive else "No TB signal (screen-negative)"
        out += [f"{MODELS[key][0]}\nP(TB) = {prob:.2f}  |  threshold = {thr:.2f}\n-> {verdict}", overlay]
    return tuple(out)


with gr.Blocks(title="TB chest X-ray screening - research demo") as demo:
    gr.Markdown("# TB chest X-ray screening: baseline vs. LoRA-fine-tuned DINOv2\n" + DISCLAIMER)
    inp = gr.Image(type="pil", label="Chest X-ray (PNG/JPG)")
    btn = gr.Button("Analyse", variant="primary")
    with gr.Row():
        with gr.Column():
            t1, i1 = gr.Textbox(label=MODELS["baseline"][0], lines=3), gr.Image(label="Grad-CAM")
        with gr.Column():
            t2, i2 = gr.Textbox(label=MODELS["proposed"][0], lines=3), gr.Image(label="Grad-CAM")
    btn.click(run, inp, [t1, i1, t2, i2])

if __name__ == "__main__":
    missing = [str(p) for _, p in MODELS.values() if not Path(p).exists()]
    if missing:
        raise SystemExit(f"Missing checkpoints: {missing}. Run: python train.py --protocol final ...")
    demo.launch()
