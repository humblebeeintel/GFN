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
from utils import resize_and_pad, to_tensor, to_image, normalize, denormalize
from utils import window_resize, capture_video, save_frame, get_gallery_values
from utils import display_images_with_labels, detect_and_highlight


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

video_file = st.file_uploader("Upload a video", type=[
                              "mp4", "avi", "mov", "mkv"])

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    st.session_state.cap = capture_video(video_path)

    if st.button("Play Video"):
        st.session_state.playing = True

    if st.button("Stop Video"):
        st.session_state.playing = False
        st.session_state.frame = st.session_state.current_frame

    cap = st.session_state.cap
    frame_placeholder = st.empty()

    while st.session_state.playing and cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            st.warning("End of video")
            st.session_state.playing = False
            break

        st.session_state.current_frame = frame
        frame_placeholder.image(frame, channels="BGR")

        time.sleep(1 / 30)

    if st.session_state.frame is not None:
        st.session_state.playing = False
        save_frame(st.session_state.frame)

if os.path.exists('current_frame.jpg'):
    img = Image.open('current_frame.jpg')
    gallery_detect_current, gallery_tensor_current = get_gallery_values(
        img, model, device)
    image = to_image(gallery_tensor_current)

    crops = []
    for i, box in enumerate(gallery_detect_current['det_boxes'].cpu().tolist()):
        x1, y1, x2, y2 = map(int, box)
        width = x2 - x1
        height = y2 - y1

        expansion_x = width * 0.30
        expansion_y = height * 0.30

        new_x1 = max(0, x1 - expansion_x)
        new_y1 = max(0, y1 - expansion_y)
        new_x2 = min(image.width, x2 + expansion_x)
        new_y2 = min(image.height, y2 + expansion_y)
        new_x1, new_y1, new_x2, new_y2 = map(
            int, (new_x1, new_y1, new_x2, new_y2))

        crop = image.crop((new_x1, new_y1, new_x2, new_y2))
        crops.append(crop)

st.write("Wait for the video to finish processing before proceeding")

try:
    query_images = [crop.convert('RGB') for crop in crops]
    labels, image_paths_corr, submit = display_images_with_labels(query_images)
except:
    st.error("No detections and crops to display. Please stop where some detections are visible")

if submit:
    st.success("Labels submitted and saved")
    
    # For debugging
    print(labels)
    print(image_paths_corr)
    
    # Save the labels and image paths to the query folder
    os.makedirs("query", exist_ok=True)
    for i, image in enumerate(image_paths_corr):
        image.save(f"query/{labels[i]}.jpg")
    
    try:
        detect_and_highlight(video_path, image_paths_corr,
                            'videos/output.mp4', model, device, labels, frame_placeholder=frame_placeholder)
    except:
        st.warning("No video to process. Please upload a video")


# RTSP CAMERA STREAM
camera_stream_url = st.text_input("RTSP camera stream URL")
if camera_stream_url:
    try:
        cap = cv2.VideoCapture(camera_stream_url)
    except:
        st.error("Failed to connect to the camera. Please check the URL")
        st.stop()
    st.session_state.playing = True
    frame_placeholder = st.empty()
    
    # Model here to detect crimes and crop the image to query
    

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture frame from camera")
            break

        frame_placeholder.image(frame, channels="BGR")
        time.sleep(1 / 30)
    cap.release()



# WEB-CAM STREAM
if st.button("Start Webcam"):
    try:
        cap = cv2.VideoCapture(0)  # 0 is the default camera
    except: 
        st.error("Failed to connect to the camera. Please check if the camera is connected properly")
    
    st.session_state.playing = True
    frame_placeholder = st.empty()
    
    
    # Model here to detect crimes and crop the image to query
    
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture frame from webcam")
            break
        
        frame_placeholder.image(frame, channels="BGR")
        time.sleep(1 / 30)
    cap.release()
