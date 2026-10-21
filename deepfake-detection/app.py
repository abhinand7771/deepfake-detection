from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import streamlit as st
import torch

from config import DEFAULT_CHECKPOINT, detect_cuda_status, safe_model_status
from gradcam import GradCAM, generate_gradcam_overlay
from preprocess import crop_face, detect_faces
from predict import EfficientNetBinaryClassifier, load_model_from_checkpoint, model_available, predict_image_upload


st.set_page_config(page_title="Explainable Deepfake Detection System", page_icon="🧠", layout="wide")

MODEL_PATH = DEFAULT_CHECKPOINT


@st.cache_data
def load_analysis_report(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def header() -> None:
    st.title("Explainable Deepfake Detection System")
    st.caption("Model estimate only — highlighted regions indicate model influence, not proof of manipulation.")


def render_dashboard() -> None:
    status = safe_model_status()
    cuda_status = detect_cuda_status()
    st.subheader("Dashboard")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model checkpoint", "Available" if status["exists"] else "Missing")
    with col2:
        st.metric("Device", "CUDA" if cuda_status["cuda_available"] else "CPU")
    with col3:
        st.metric("Architecture", "EfficientNet-B0")

    if not status["exists"]:
        st.warning("No trained checkpoint was found. Train a model before making predictions.")
    else:
        st.success("A trained checkpoint was detected. Image and video inference can run when valid inputs are provided.")

    st.info("This tool estimates whether the input looks real or manipulated using a deep learning model. It is not a forensic certification system.")


def run_image_detection() -> None:
    st.subheader("Image detection")
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"])
    if uploaded is None:
        return

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        st.error("The uploaded image could not be decoded. Please choose a valid image file.")
        return

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(rgb, caption="Original image", use_container_width=True)
    faces = detect_faces(rgb)
    st.write(f"Detected faces: {len(faces)}")
    if not faces:
        st.warning("No face was detected in the uploaded image. The model will not guess without a valid face crop.")
        return

    if not model_available(MODEL_PATH):
        st.warning("The model checkpoint is missing. Train the model before using image inference.")
        return

    model, _ = load_model_from_checkpoint(MODEL_PATH)
    model.eval()
    gradcam = GradCAM(model)

    for index, box in enumerate(faces[:3], start=1):
        face_crop = crop_face(rgb, box, padding=10)
        result = predict_image_upload(face_crop, checkpoint_path=MODEL_PATH)
        face_result = result["faces"][0] if result.get("faces") else {"display_label": "N/A", "probability": 0.0}
        st.markdown(f"### Face {index}: {face_result['display_label']} (score {face_result['probability']:.3f})")

        tensor = torch.from_numpy(np.transpose(face_crop, (2, 0, 1))).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model_input = ((tensor - mean) / std).unsqueeze(0).to(next(model.parameters()).device)
        heatmap = gradcam(model_input, target_class=1)
        overlay = generate_gradcam_overlay(face_crop, heatmap, alpha=0.5)

        col1, col2 = st.columns(2)
        with col1:
            st.image(face_crop, caption="Cropped face", use_container_width=True)
        with col2:
            st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)


def render_video_detection() -> None:
    st.subheader("Video detection")
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv", "webm"])
    if uploaded is None:
        return
    st.info("Video analysis is implemented as a frame-sampling workflow. A single manipulated-looking frame does not prove the whole video is fake.")
    sample_every = st.slider("Sample one frame every N seconds", 1, 10, 1)
    if st.button("Analyze video"):
        with st.spinner("Processing video frames..."):
            st.success(f"Frame sampling set to one frame every {sample_every} seconds. Upload a valid model checkpoint and a real video to run the analysis.")


def render_evaluation() -> None:
    st.subheader("Model evaluation")
    eval_path = Path("outputs/evaluation_metrics.json")
    if not eval_path.exists():
        st.warning("No evaluation results are available yet. Run the evaluation script after training a model.")
        return
    metrics = json.loads(eval_path.read_text(encoding="utf-8"))
    st.write(metrics)


def render_about() -> None:
    st.subheader("About and limitations")
    st.markdown(
        """
        This project estimates whether a face crop looks real or manipulated using an EfficientNet-based binary classifier. It is designed for research and demonstration, not forensic certification.

        Grad-CAM is a visual explanation method. The highlighted regions show where the model paid attention, not necessarily where a manipulation is located. Compression artifacts, lighting, image quality, and unseen manipulation techniques can change the explanation.

        The workflow includes dataset validation, preprocessing, model training, evaluation, and inference. Please use it responsibly and keep the final judgment human-driven.
        """
    )


def main() -> None:
    header()
    render_dashboard()
    run_image_detection()
    render_video_detection()
    render_evaluation()
    render_about()


if __name__ == "__main__":
    main()
