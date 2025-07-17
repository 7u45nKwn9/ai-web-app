import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import onnxruntime as ort
import yaml
from PIL import Image
from utils.yolo_utils import (
    preprocess,
    postprocess_yolov8_raw,
    draw_boxes,
    process_image,
    non_max_suppression
)

# Load class names
with open("model/data.yaml", "r") as f:
    class_names = yaml.safe_load(f)["names"]

# Load model ONNX
MODEL_PATH = "model/polyp.onnx"
MODEL_RESOLUTION = (640, 640)
session = ort.InferenceSession(MODEL_PATH)

# Streamlit UI
st.set_page_config(page_title="Model Web App", layout="centered")
st.title("Object Detection Web App")
st.markdown("Supports both 📸 Image Capture and 🔴 Live Detection")

# Mode selection
mode = st.radio("Choose Mode", ["📸 Import Image", "🔴 Live Detection"])

# ----------------------------
# 🔴 LIVE DETECTION SECTION
# ----------------------------
if mode == "🔴 Live Detection":
    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    cap = None
    if run:
        cap = cv2.VideoCapture(0)

    while run and cap and cap.isOpened():
        success, frame = cap.read()
        if not success:
            st.write("Failed to grab frame")
            break

        input_tensor = preprocess(cv2.resize(frame, MODEL_RESOLUTION))
        outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
        boxes = postprocess_yolov8_raw(outputs, frame.shape[:2], MODEL_RESOLUTION)
        boxes = non_max_suppression(boxes)

        frame = draw_boxes(frame, boxes, class_names)

        FRAME_WINDOW.image(frame, channels="BGR")

    if cap:
        cap.release()

# ----------------------------
# 📸 IMPORT IMAGE
# ----------------------------
elif mode == "📸 Import Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        # Inference
        result = process_image(
            frame=img,
            session=session,
            preprocess=preprocess,
            postprocess=postprocess_yolov8_raw,
            draw_boxes=draw_boxes,
            model_name=MODEL_PATH,
            model_resolution=MODEL_RESOLUTION,
            classes=class_names,
        )

        # Display
        st.image(result, caption="Detected Image", channels="BGR")

        # Allow download
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        is_success, buffer = cv2.imencode(".jpg", result_rgb)
        st.download_button(
            label="📥 Download Result",
            data=buffer.tobytes(),
            file_name="detection_result.jpg",
            mime="image/jpeg"
        )
