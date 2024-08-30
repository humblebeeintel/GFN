import ssl
from PIL import Image
import tempfile
import streamlit as st
import torchvision.transforms.functional as TF
from albumentations.augmentations.geometric import functional as FGeometric
import numpy as np
import torch.nn.functional as F
import cv2
import os
from torchvision import transforms
import torch
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_nvfuser_enabled(False)
import time
ssl._create_default_https_context = ssl._create_unverified_context

# Convert PIL image to torch tensor
def to_tensor(image, device='cpu'):
    arr = np.array(image)
    arr_wrs = window_resize(arr)
    tsr = torch.FloatTensor(arr_wrs)
    tsr_norm = normalize(tsr)
    tsr_input = tsr_norm.permute(2, 0, 1).to(device)
    return tsr_input

# Convert torch tensor to PIL image
def to_image(tensor):
    tsr_denorm = denormalize(tensor.permute(1, 2, 0).cpu()).clip(min=0, max=1)
    arr = tsr_denorm.numpy()
    arr_uint8 = (arr * 255.0).astype(np.uint8)
    image = Image.fromarray(arr_uint8)
    return image

# Normalize image tensor using ImageNet stats
def normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.FloatTensor(mean).view(1, 1, 3)
    std = torch.FloatTensor(std).view(1, 1, 3)
    return tensor.div(255.0).sub(mean).div(std)

# Denormalize image tensor using ImageNet stats
def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.FloatTensor(mean).view(1, 1, 3)
    std = torch.FloatTensor(std).view(1, 1, 3)
    return tensor.mul(std).add(mean)

# Resize image (numpy array) to fit in fixed size window
def window_resize(img, min_size=900, max_size=1500, interpolation=cv2.INTER_LINEAR):
    height, width = img.shape[:2]
    image_min_size = min(width, height)
    image_max_size = max(width, height)
    scale_factor = min_size / image_min_size
    if image_max_size * scale_factor > max_size:
        return FGeometric.longest_max_size(img, max_size=max_size, interpolation=interpolation)
    else:
        return FGeometric.smallest_max_size(img, max_size=min_size, interpolation=interpolation)

def get_query_values(image, model, device):
    query_tensor = to_tensor(image)
    query_box = torch.FloatTensor([0, 0, *query_tensor.shape[1:]]).unsqueeze(0).to(device)
    query_targets = [{'boxes': query_box}]
    with torch.no_grad():
        detections = model([query_tensor], query_targets, inference_mode='both')
    query_detect = detections[0]
    return query_detect

def get_gallery_values(frame, model, device):
    frame_tensor = to_tensor(frame).to(device)
    with torch.no_grad():
        detections_list = model([frame_tensor], inference_mode='det')
    return detections_list[0], frame_tensor

def detect_and_highlight(video_path, query_images, output_path, model, device, labels):
    embeddings = []
    for image in query_images:
        query_detect = get_query_values(image, model, device)
        embeddings.append(query_detect['det_emb'])
        print(embeddings)
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    progress_bar = "Operation in progress. Please wait..."
    my_bar = st.progress(0, text=progress_bar) 
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        print(f"Processing frame {frame_num}/{frame_count}")

        try:  
            time.sleep(0.01)
            my_bar.progress(frame_num / frame_count, text=progress_bar)
        except:
            pass

        frame = cv2.resize(frame, (1200, 900))
        gallery_detect, _ = get_gallery_values(
            frame, model, device)
        
        person_sim_query = {}
        b = 0
        for emb in embeddings:
            person_sim = torch.mm(
                F.normalize(emb, dim=1),
                F.normalize(gallery_detect['det_emb'], dim=1).T
            ).flatten()
            person_sim_query[labels[b]] = person_sim
            b += 1
        
        print(person_sim_query)

        max_sim_per_detection = torch.full((len(gallery_detect['det_boxes']),), -1.0)
        assigned_person_for_detection = [-1] * len(gallery_detect['det_boxes'])
        best_detection_for_person = {}

        # Find the best matching detection for each person
        for label, person_sim in person_sim_query.items():
            best_sim, best_detection = torch.max(person_sim, 0)
            if best_sim > 0.3:
                if best_detection_for_person.get(best_detection.item()) is None:
                    best_detection_for_person[best_detection.item()] = (label, best_sim.item())
                else:
                    existing_label, existing_sim = best_detection_for_person[best_detection.item()]
                    if best_sim.item() > existing_sim:
                        best_detection_for_person[best_detection.item()] = (label, best_sim.item())

        final_assignments = {}
        for detection, (label, sim_score) in best_detection_for_person.items():
            if label not in final_assignments:
                final_assignments[label] = detection

        for label, detection in final_assignments.items():
            box = gallery_detect['det_boxes'][detection]
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        out.write(frame)
        st.image(frame, channels="BGR")
        save_frame(frame, filename=f"frames/frame{frame_num}.jpg")  
    time.sleep(1)
    my_bar.empty()
    cap.release()
    out.release()

def display_images_with_labels(image_paths):
    labels = []
    image_paths_corr = []

    # Calculate the number of columns needed
    num_images = len(image_paths)
    num_cols = min(num_images, 3)
    # Dynamically create rows of images
    for i in range(0, num_images, num_cols):
        cols = st.columns(num_cols)
        for idx, col in enumerate(cols):
            if i + idx < num_images:
                with col:
                    img_path = image_paths[i + idx]
                    
                    img = img_path
                    img = img.resize((20, 40))

                    st.image(img, use_column_width=True)
                    # Get the label from the user, if not provided use the default
                    agree = st.checkbox(f"Select Image {i + idx + 1}")
                    if agree:
                        label = st.text_input(f"Label for Image {i + idx + 1}" , key=f"label_{i + idx}")
                        labels.append(label)
                        image_paths_corr.append(img_path)
    
    if st.button("Submit All Labels"):
        return labels, image_paths_corr,  True
    return labels, image_paths_corr, False

def capture_video(video_path):
    cap = cv2.VideoCapture(video_path)
    return cap

def display_frame(frame):
    st.image(frame, channels="RGB")

def save_frame(frame, filename="current_frame.jpg"):
    cv2.imwrite(filename, frame)
    st.success(f"Frame saved as {filename}")

# Start

st.title("Video Query Detection App")

# File uploaders

device = torch.device('cpu')
model = torch.jit.load('cuhk_final_convnext-base_e30.torchscript.pt', map_location=device)

if 'playing' not in st.session_state:
    st.session_state.playing = False
if 'frame' not in st.session_state:
    st.session_state.frame = None

if 'crop_index' not in st.session_state:
    st.session_state.crop_index = 0  # Track the current crop index
if 'crop_labels' not in st.session_state:
    st.session_state.crop_labels = []  # Store user labels

video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

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
    gallery_detect_current, gallery_tensor_current = get_gallery_values(img, model, device)
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
        new_x1, new_y1, new_x2, new_y2 = map(int, (new_x1, new_y1, new_x2, new_y2))

        crop = image.crop((new_x1, new_y1, new_x2, new_y2))
        crops.append(crop)

st.write("Wait for the video to finish processing before proceeding")

query_images = [crop.convert('RGB') for crop in crops]
labels, image_paths_corr, submit = display_images_with_labels(query_images)

if submit:
    st.success("Labels submitted")
    print(labels)
    print(image_paths_corr)

    detect_and_highlight(video_path, image_paths_corr, 'output.mp4', model, device, labels)
