import os
from typing import List, Tuple, Dict

import gradio as gr
import torch
import numpy as np
from PIL import Image

import config
from src import create_model, get_transforms, load_data


model = None
label_encoder = None
transform = None
device = None
all_classes = []


def load_classifier():
    global model, label_encoder, transform, device, all_classes

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = (
        config.BEST_MODEL_PATH
        if os.path.exists(config.BEST_MODEL_PATH)
        else config.MODEL_PATH
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please train the model first by running: python main.py"
        )

    if not os.path.exists(config.DATA_PATH):
        raise FileNotFoundError(
            f"Data not found at {config.DATA_PATH}. "
            "Please download and consolidate datasets first."
        )

    df, label_encoder = load_data(config.DATA_PATH)
    num_classes = len(df["labels"].unique())
    all_classes = list(label_encoder.classes_)

    model = create_model(
        config.MODEL_NAME, num_classes, pretrained=False, dropout=0.0
    ).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    transform = get_transforms(config.IMG_SIZE, augment=False)

    return num_classes


def get_predictions(image_tensor, top_k: int = 5) -> List[Tuple[str, float]]:
    with torch.no_grad():
        image_tensor = image_tensor.to(device).unsqueeze(0)
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)[0]

        top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))

        results = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            class_name = label_encoder.inverse_transform([idx])[0]
            results.append((class_name, float(prob)))

    return results


def classify_image(image: Image.Image, top_k: int):
    if image is None:
        return {}, "No image provided"

    image_rgb = image.convert("RGB")
    image_tensor = transform(image_rgb)

    predictions = get_predictions(image_tensor, top_k)

    class_probs = {cls: prob for cls, prob in predictions}

    for cls in all_classes:
        if cls not in class_probs:
            class_probs[cls] = 0.0

    result_text = "**Top Predictions:**\n\n"
    for cls, prob in predictions:
        bar_len = int(prob * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        result_text += f"**{cls}**: {bar} {prob:.1%}\n\n"

    return class_probs, result_text


css = """
.gradio-container {max-width: 900px !important; margin: auto !important;}
.prediction-label {font-size: 20px !important; font-weight: bold !important;}
.confidence-bar {height: 24px !important; border-radius: 4px;}
.result-box {padding: 15px; border-radius: 10px; background: #f8f9fa;}
"""


try:
    num_classes = load_classifier()
    print(f"Model loaded successfully on {device}")
    print(f"Number of classes: {num_classes}")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("\nTo fix this:")
    print("1. Consolidate datasets: python scripts/consolidate_datasets.py")
    print("2. Train model: python main.py")
    exit(1)


def create_demo():
    with gr.Blocks(css=css, title="Animal Classifier") as demo:
        gr.Markdown("# Animal Classifier")
        gr.Markdown(
            f"Classifies images into **{num_classes}** animal categories. "
            "Upload an image to see top predictions with confidence scores."
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Image")
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=min(10, num_classes),
                    value=5,
                    step=1,
                    label="Show Top K Predictions",
                )
                classify_btn = gr.Button("Classify", variant="primary", size="lg")

            with gr.Column(scale=1):
                prediction_output = gr.Label(
                    num_top_classes=config.TOP_K_PREDICTIONS, label="Predictions"
                )
                result_text = gr.Markdown(
                    value="*Upload an image and click Classify to see predictions*",
                    elem_classes=["result-box"],
                )

        classify_btn.click(
            fn=classify_image,
            inputs=[image_input, top_k_slider],
            outputs=[prediction_output, result_text],
        )

        gr.Markdown(
            f"""
            ---
            **Model Info:**
            - Device: {device}
            - Model: {config.MODEL_NAME}
            - Classes: {num_classes}
            - Image Size: {config.IMG_SIZE}x{config.IMG_SIZE}
            """
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
