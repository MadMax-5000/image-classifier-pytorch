import os
import gradio as gr
import torch
from PIL import Image

import config
from src import create_model, get_transforms, load_data


def load_classifier():
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
            "Please download the dataset first by running: python scripts/download_data.py"
        )

    df, label_encoder = load_data(config.DATA_PATH)
    num_classes = len(df["labels"].unique())

    model = create_model(config.MODEL_NAME, num_classes, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    transform = get_transforms(config.IMG_SIZE, augment=False)

    return model, label_encoder, transform, device


def classify_image(image: Image.Image):
    if image is None:
        return {}, "No image provided"

    image_rgb = image.convert("RGB")
    image_tensor = transform(image_rgb)

    with torch.no_grad():
        image_tensor = image_tensor.to(device).unsqueeze(0)
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    class_probs = {
        label: float(prob)
        for label, prob in zip(label_encoder.classes_, probs.cpu().numpy())
    }

    prediction = label_encoder.inverse_transform([torch.argmax(probs, dim=0).item()])[0]

    return class_probs, f"Prediction: {prediction}"


css = """
.gradio-container {max-width: 800px !important; margin: auto !important;}
.prediction-label {font-size: 24px !important; font-weight: bold !important; text-align: center !important;}
.confidence-bar {height: 30px !important;}
"""


try:
    model, label_encoder, transform, device = load_classifier()
    print(f"Model loaded successfully on {device}")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("\nTo fix this:")
    print("1. Download data: python scripts/download_data.py")
    print("2. Train model: python main.py")
    exit(1)


def create_demo():
    with gr.Blocks(css=css, title="Animal Face Classifier") as demo:
        gr.Markdown("# Animal Face Classifier")
        gr.Markdown(
            "Upload an image to classify it as **cat**, **dog**, or **wild** animal"
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")
                classify_btn = gr.Button("Classify", variant="primary")

            with gr.Column():
                prediction_output = gr.Label(num_top_classes=3, label="Prediction")
                prediction_text = gr.Textbox(
                    label="Result",
                    lines=1,
                    interactive=False,
                    elem_classes=["prediction-label"],
                )

        classify_btn.click(
            fn=classify_image,
            inputs=[image_input],
            outputs=[prediction_output, prediction_text],
        )

        gr.Markdown(
            f"""
            ---
            **Model Info:**
            - Device: {device}
            - Model: {config.MODEL_NAME}
            - Classes: {", ".join(label_encoder.classes_)}
            - Image Size: {config.IMG_SIZE}x{config.IMG_SIZE}
            """
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
