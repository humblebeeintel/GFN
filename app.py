import ssl
from PIL import Image, ImageOps
import os
import torchvision.transforms.functional as TF
from albumentations.augmentations.geometric import functional as FGeometric
import numpy as np
import cv2
import time
import tempfile
import torch.nn.functional as F
import torch
import streamlit as st

torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_nvfuser_enabled(False)

ssl._create_default_https_context = ssl._create_unverified_context

# Set up the Streamlit page
st.set_page_config(layout="wide")
st.title("Person Re-Identification System")

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.jit.load('torchscripts/cuhk_final_convnext-base_e30.torchscript.pt', map_location=device)

# Initialize session state
if 'playing' not in st.session_state:
    st.session_state.playing = False
if 'frame' not in st.session_state:
    st.session_state.frame = None
if 'crop_index' not in st.session_state:
    st.session_state.crop_index = 0
if 'crop_labels' not in st.session_state:
    st.session_state.crop_labels = []
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'stop_video' not in st.session_state:
    st.session_state.stop_video = False
if 'gallery_detect' not in st.session_state:
    st.session_state.gallery_detect = None
if 'captured_crops' not in st.session_state:
    st.session_state.captured_crops = []
if 'query_images' not in st.session_state:
    st.session_state.query_images = []

# Helper functions

def process_uploaded_videoname(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    return tfile.name

def resize_and_pad(img, target_height, target_width):
    aspect_ratio = img.width / img.height
    new_width = int(target_height * aspect_ratio)
    img_resized = img.resize((new_width, target_height), Image.Resampling.LANCZOS)
    delta_w = target_width - new_width
    padding = (delta_w // 2, 0, delta_w - (delta_w // 2), 0)
    img_padded = ImageOps.expand(img_resized, padding, fill=(255, 255, 255))
    return img_padded

def to_tensor(image, device='cuda'):
    arr = np.array(image)
    arr_resized = window_resize(arr)
    tsr = torch.FloatTensor(arr_resized)
    tsr_norm = normalize(tsr)
    tsr_input = tsr_norm.permute(2, 0, 1).to(device)
    return tsr_input

def normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.FloatTensor(mean).view(1, 1, 3)
    std = torch.FloatTensor(std).view(1, 1, 3)
    return tensor.div(255.0).sub(mean).div(std)

def window_resize(img, min_size=900, max_size=1500, interpolation=cv2.INTER_LINEAR):
    height, width = img.shape[:2]
    scale_factor = min_size / min(width, height)
    if max(width, height) * scale_factor > max_size:
        return FGeometric.longest_max_size(img, max_size=max_size, interpolation=interpolation)
    else:
        return FGeometric.smallest_max_size(img, max_size=min_size, interpolation=interpolation)

def get_gallery_values(frame, model, device):
    frame_tensor = to_tensor(frame).to(device)
    with torch.no_grad():
        detections_list = model([frame_tensor], inference_mode='det')
    return detections_list[0], frame_tensor

def find_best_detections(person_sim_query):
    best_detection_for_person = {}
    for label, person_sim in person_sim_query.items():
        try:
            best_sim, best_detection = torch.max(person_sim, 0)
        except:
            break
        if best_sim > 0.2:
            detection_id = best_detection.item()
            if detection_id not in best_detection_for_person or best_sim.item() > best_detection_for_person[detection_id][1]:
                best_detection_for_person[detection_id] = (label, best_sim.item())
    return best_detection_for_person

def annotate_frame_with_detections(frame, final_assignments, gallery_detect, labels):
    for index, detection in final_assignments.items():
        box = gallery_detect['det_boxes'][index]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label, score = detection
        cv2.putText(frame, f'{label} : {round(score, 2) * 100}%', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

def load_db_embeddings():
    labels = []
    embeddings = []
    for file in os.listdir("query_crops"):
        if file.endswith(".npy"):
            label = file.split(".")[0]
            emb = torch.from_numpy(np.load(f"query_crops/{file}")).to(device)
            labels.append(label)
            embeddings.append(emb)
    return labels, torch.stack(embeddings).to(device)

def display_query_crops():
    st.sidebar.header("Query Crops")
    query_crop_dir = "query_crops"
    if os.path.exists(query_crop_dir):
        if not st.session_state.query_images:
            st.session_state.query_images = [img for img in os.listdir(query_crop_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
        
        num_cols = min(len(st.session_state.query_images), 3)
        cols = st.sidebar.columns(num_cols)
        for idx, img_name in enumerate(st.session_state.query_images):
            img_path = os.path.join(query_crop_dir, img_name)
            img = Image.open(img_path)
            img_resized_padded = resize_and_pad(img, 150, 150)
            cols[idx % num_cols].image(img_resized_padded, caption=img_name, use_column_width=True)
    else:
        st.sidebar.write("No query crops available.")

def capture_crop(frame, gallery_detect):
    for i, box in enumerate(gallery_detect['det_boxes']):
        x1, y1, x2, y2 = map(int, box)
        crop_img = frame[y1:y2, x1:x2]
        crop_img_small = cv2.resize(crop_img, (150, 150))
        st.session_state.captured_crops.append((crop_img, crop_img_small, i))

def save_crop(crop_img, crop_name, index):
    query_crop_dir = "query_crops"
    if not os.path.exists(query_crop_dir):
        os.makedirs(query_crop_dir)
    crop_path = os.path.join(query_crop_dir, f"{crop_name}.png")
    cv2.imwrite(crop_path, crop_img)
    embedding = st.session_state.gallery_detect['det_emb'][index].cpu().detach().numpy().reshape(1, 2048)
    embedding_path = os.path.join(query_crop_dir, f"{crop_name}.npy")
    np.save(embedding_path, embedding)
    st.sidebar.write(f"Crop and embedding saved as {crop_name}")
    st.session_state.query_images.append(f"{crop_name}.png")

def main():
    video_source = st.sidebar.selectbox("Select Video Source", ("Upload Video", "Webcam"))

    if video_source == "Upload Video":
        video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
        if video_file:
            st.session_state.video_path = process_uploaded_videoname(video_file)
    elif video_source == "Webcam":
        st.session_state.video_path = 0  # 0 is the default webcam

    labels, embeddings = load_db_embeddings()

    if st.sidebar.button("Show Query Crops"):
        display_query_crops()
    if st.sidebar.button("Start Video Feed"):
        st.session_state.stop_video = False
    if st.sidebar.button("Stop Video Feed"):
        st.session_state.stop_video = True
    if st.sidebar.button("Capture Crop"):
        if st.session_state.frame is not None and st.session_state.gallery_detect is not None:
            capture_crop(st.session_state.frame, st.session_state.gallery_detect)
        else:
            st.sidebar.warning("No frame or detection available to capture crop.")

    crops_to_remove = []
    for i, (crop_img, crop_img_small, index) in enumerate(st.session_state.captured_crops):
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
        st.sidebar.image(crop_img_small, caption=f"Captured Crop {i+1}", use_column_width=True, channels="BGR")
        crop_name = st.sidebar.text_input(f"Crop Name {i+1}", key=f"crop_name_{i}")
        submit = st.sidebar.button(f"Save Crop {i+1}")
        
        if crop_name and submit:
            save_crop(crop_img, crop_name, index)
            crops_to_remove.append(i)

    for i in sorted(crops_to_remove, reverse=True):
        st.session_state.captured_crops.pop(i)

    if st.session_state.video_path is not None and not st.session_state.stop_video:
        feed_placeholder = st.empty()
        cap = cv2.VideoCapture(st.session_state.video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if st.session_state.video_path != 0 else None
        my_bar = st.progress(0, text="Operation in progress. Please wait...") if frame_count else None
        tick = time.time()
        fps = 0
        frame_num = 0

        while cap.isOpened():
            if st.session_state.stop_video:
                break
            ret, frame = cap.read()
            if not ret:
                st.warning("End of video")
                break

            if frame_count:
                my_bar.progress((frame_num + 1) / frame_count, text=f"Processing frame {frame_num + 1} of {frame_count}...")

            frame = cv2.resize(frame, (1200, 900))
            gallery_detect, _ = get_gallery_values(frame, model, device)
            st.session_state.gallery_detect = gallery_detect
            person_sim_query = {
                labels[b]: torch.mm(F.normalize(emb, dim=1), F.normalize(gallery_detect['det_emb'], dim=1).T).flatten()
                for b, emb in enumerate(embeddings)
            }

            best_detection_for_person = find_best_detections(person_sim_query)
            annotate_frame_with_detections(frame, best_detection_for_person, gallery_detect, labels)
            st.session_state.frame = frame
            cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Frame ID: {frame_num + 1}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            feed_placeholder.image(frame, channels="BGR")
            tock = time.time()
            fps = round(1 / (tock - tick), 2)
            tick = tock
            frame_num += 1
        if my_bar:
            my_bar.empty()
        cap.release()

if __name__ == "__main__":
    main()
