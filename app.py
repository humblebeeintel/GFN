from utils import display_images_with_labels, detect_and_highlight
from utils import window_resize, capture_video, save_frame, get_gallery_values
from utils import resize_and_pad, to_tensor, to_image, normalize, denormalize
import time
import ssl
from PIL import Image, ImageOps
import tempfile
import streamlit as st
import torchvision.transforms.functional as TF
from albumentations.augmentations.geometric import functional as FGeometric
import numpy as np
import torch.nn.functional as F
import cv2
import os
import torch
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_nvfuser_enabled(False)
ssl._create_default_https_context = ssl._create_unverified_context

# Import functions from utils.py


# Start
st.title("Video Query Detection App")

# File uploaders
device = torch.device('cuda')
model = torch.jit.load(
    'cuhk_final_convnext-base_e30.torchscript.pt', map_location=device)

if 'playing' not in st.session_state:
    st.session_state.playing = False
if 'frame' not in st.session_state:
    st.session_state.frame = None

if 'crop_index' not in st.session_state:
    st.session_state.crop_index = 0  # Track the current crop index
if 'crop_labels' not in st.session_state:
    st.session_state.crop_labels = []  # Store user labels


def process_uploaded_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    return tfile.name


def play_video(cap, frame_placeholder):
    while st.session_state.playing and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("End of video")
            st.session_state.playing = False
            break
        st.session_state.current_frame = frame
        frame_placeholder.image(frame, channels="BGR")
        time.sleep(1 / 30)


def save_current_frame():
    if st.session_state.frame is not None:
        save_frame(st.session_state.frame)


def process_current_frame(img, model, device):
    gallery_detect_current, gallery_tensor_current = get_gallery_values(
        img, model, device)
    image = to_image(gallery_tensor_current)

    crops = []
    for box in gallery_detect_current['det_boxes'].cpu().tolist():
        crop = crop_image_with_expansion(image, box)
        crops.append(crop)

    return crops


def crop_image_with_expansion(image, box, expansion_ratio=0.20):
    x1, y1, x2, y2 = map(int, box)
    width, height = x2 - x1, y2 - y1

    expansion_x = width * expansion_ratio
    expansion_y = height * expansion_ratio

    new_x1 = max(0, x1 - expansion_x)
    new_y1 = max(0, y1 - expansion_y)
    new_x2 = min(image.width, x2 + expansion_x)
    new_y2 = min(image.height, y2 + expansion_y)

    return image.crop((int(new_x1), int(new_y1), int(new_x2), int(new_y2)))


def process_and_display_labels(crops, model):
    query_images = [crop.convert('RGB') for crop in crops]
    labels, embeddings, submit = display_images_with_labels(
        query_images, model)
    return labels, embeddings, submit


def save_embeddings(labels, embeddings):
    os.makedirs("query", exist_ok=True)
    for label, emb in zip(labels, embeddings):
        # Save embedings as numpy array
        np.save(f"query/{label}.npy", emb.cpu().numpy())


def start_webcam_stream(frame_placeholder):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Failed to connect to the camera.")
        return

    st.session_state.playing = True
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture frame from webcam")
            break
        frame_placeholder.image(frame, channels="BGR")
        time.sleep(1 / 30)
    cap.release()


def main():
    video_file = st.file_uploader("Upload a video", type=[
                                  "mp4", "avi", "mov", "mkv"])

    if video_file is not None:
        video_path = process_uploaded_video(video_file)
        st.session_state.cap = capture_video(video_path)

        if st.button("Play Video"):
            st.session_state.playing = True

        if st.button("Label Identities: Kishilarni qayd etish"):
            st.session_state.playing = False
            st.session_state.frame = st.session_state.current_frame

        frame_placeholder = st.empty()
        play_video(st.session_state.cap, frame_placeholder)
        save_current_frame()

    if os.path.exists('current_frame.jpg'):
        img = Image.open('current_frame.jpg')
        crops = process_current_frame(img, model, device)

        try:
            labels, embeddings, submit = process_and_display_labels(
                crops, model)
        except:
            st.error("No detections and crops to display.")

        if submit:
            st.success("Labels submitted and saved")
            save_embeddings(labels, embeddings)

            try:
                detect_and_highlight(
                    video_path, embeddings, 'videos/output.mp4', model, device, labels, frame_placeholder)
            except:
                st.warning("No video to process. Please upload a video")

    if st.button("Start Webcam"):
        frame_placeholder = st.empty()
        start_webcam_stream(frame_placeholder)


if __name__ == "__main__":
    main()
