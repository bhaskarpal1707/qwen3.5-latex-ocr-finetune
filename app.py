import json
import os
import torch
import gradio as gr
from PIL import Image

MODEL_ID = "bhaskar1707/qwen3.5-latex-ocr-finetune"

# ZeroGPU decorator — gives free shared A100 GPU per request
import spaces

@spaces.GPU
def load_model_and_generate(image, instruction="Write the LaTeX representation for this image."):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        trust_remote_code=True,
    ).to("cuda")
    model.eval()

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128, use_cache=True, temperature=1.5, min_p=0.1)

    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


# ── Tab 1: Try it ──────────────────────────────────────────────────────────────
def predict(image):
    if image is None:
        return "Please upload an image.", ""
    try:
        pil_image = Image.fromarray(image).convert("RGB")
        latex = load_model_and_generate(pil_image)
        return latex, f"$${latex}$$"
    except Exception as e:
        return f"Error: {str(e)}", ""


with gr.Blocks(title="LaTeX OCR — Qwen3.5 Fine-tuned") as demo:
    gr.Markdown("# 🧮 Handwritten Formula → LaTeX\nFine-tuned Qwen3.5-0.8B on the LaTeX OCR dataset.")

    with gr.Tab("Try it"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Upload formula image")
                run_btn = gr.Button("Generate LaTeX", variant="primary")
            with gr.Column():
                latex_code = gr.Textbox(label="Predicted LaTeX code", lines=4)
                latex_render = gr.Markdown(label="Rendered formula")

        run_btn.click(fn=predict, inputs=image_input, outputs=[latex_code, latex_render])

    with gr.Tab("Model Performance"):
        gr.Markdown("## Before vs. After Fine-tuning")

        if os.path.exists("metrics.json"):
            with open("metrics.json") as f:
                metrics = json.load(f)
            before = metrics["before_finetuning"]
            after  = metrics["after_finetuning"]

            gr.Markdown(f"""
| Metric | Before | After |
|---|---|---|
| Mean CER ↓ | {before['mean_cer']:.3f} | {after['mean_cer']:.3f} |
| Exact Match ↑ | {before['exact_match_accuracy']:.1%} | {after['exact_match_accuracy']:.1%} |
| Mean BLEU ↑ | {before['mean_bleu']:.3f} | {after['mean_bleu']:.3f} |
""")
        else:
            gr.Markdown("_metrics.json not found_")

        if os.path.exists("before_after_metrics.png"):
            gr.Image("before_after_metrics.png", label="Before vs After Metrics")
        if os.path.exists("training_loss_curve.png"):
            gr.Image("training_loss_curve.png", label="Training Loss Curve")
        if os.path.exists("training_time_memory.png"):
            gr.Image("training_time_memory.png", label="Training Time & GPU Memory")

demo.launch()