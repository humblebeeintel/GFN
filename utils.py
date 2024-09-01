import ssl
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
from albumentations.augmentations.geometric import functional as FGeometric
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import streamlit as st

# Disable certain torch optimizations
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_nvfuser_enabled(False)

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context


def resize_and_pad(img, target_height, target_width):
    aspect_ratio = img.width / img.height
    new_width = int(target_height * aspect_ratio)
    img_resized = img.resize((new_width, target_height),
                             Image.Resampling.LANCZOS)

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


def to_image(tensor):
    tsr_denorm = denormalize(tensor.permute(1, 2, 0).cpu()).clip(min=0, max=1)
    arr = tsr_denorm.numpy()
    arr_uint8 = (arr * 255.0).astype(np.uint8)
    image = Image.fromarray(arr_uint8)
    return image


def normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.FloatTensor(mean).view(1, 1, 3)
    std = torch.FloatTensor(std).view(1, 1, 3)
    return tensor.div(255.0).sub(mean).div(std)


def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.FloatTensor(mean).view(1, 1, 3)
    std = torch.FloatTensor(std).view(1, 1, 3)
    return tensor.mul(std).add(mean)


def window_resize(img, min_size=900, max_size=1500, interpolation=cv2.INTER_LINEAR):
    height, width = img.shape[:2]
    scale_factor = min_size / min(width, height)
    if max(width, height) * scale_factor > max_size:
        return FGeometric.longest_max_size(img, max_size=max_size, interpolation=interpolation)
    else:
        return FGeometric.smallest_max_size(img, max_size=min_size, interpolation=interpolation)


def get_query_values(image, model, device):
    query_tensor = to_tensor(image)
    query_box = torch.FloatTensor(
        [0, 0, *query_tensor.shape[1:]]).unsqueeze(0).to(device)
    query_targets = [{'boxes': query_box}]
    with torch.no_grad():
        detections = model([query_tensor], query_targets,
                           inference_mode='both')
    return detections[0]


def get_gallery_values(frame, model, device):
    frame_tensor = to_tensor(frame).to(device)
    with torch.no_grad():
        detections_list = model([frame_tensor], inference_mode='det')
    return detections_list[0], frame_tensor


def detect_and_highlight(video_path, embeddings, output_path, model, device, labels, frame_placeholder):

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    my_bar = st.progress(0, text="Operation in progress. Please wait...")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("End of video")
            break

        frame_num += 1
        my_bar.progress(frame_num / frame_count,
                        text="Operation in progress. Please wait...")

        frame = cv2.resize(frame, (1200, 900))
        gallery_detect, _ = get_gallery_values(frame, model, device)

        person_sim_query = {
            labels[b]: torch.mm(
                F.normalize(emb, dim=1),
                F.normalize(gallery_detect['det_emb'], dim=1).T
            ).flatten()
            for b, emb in enumerate(embeddings)
        }

        best_detection_for_person = find_best_detections(person_sim_query)

        final_assignments = assign_detections(best_detection_for_person)

        annotate_frame_with_detections(
            frame, final_assignments, gallery_detect, labels)

        out.write(frame)
        frame_placeholder.image(frame, channels="BGR")

    my_bar.empty()
    cap.release()
    out.release()


def find_best_detections(person_sim_query):
    best_detection_for_person = {}
    for label, person_sim in person_sim_query.items():
        best_sim, best_detection = torch.max(person_sim, 0)
        if best_sim > 0.2:
            if best_detection.item() not in best_detection_for_person or \
                    best_sim.item() > best_detection_for_person[best_detection.item()][1]:
                best_detection_for_person[best_detection.item()] = (
                    label, best_sim.item())
    return best_detection_for_person


def assign_detections(best_detection_for_person):
    final_assignments = {}
    for detection, (label, _) in best_detection_for_person.items():
        if label not in final_assignments:
            final_assignments[label] = detection
    return final_assignments


def annotate_frame_with_detections(frame, final_assignments, gallery_detect, labels):
    for label, detection in final_assignments.items():
        box = gallery_detect['det_boxes'][detection]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)


def display_images_with_labels(image_paths, model):
    labels = []
    embeddings = []

    num_images = len(image_paths)
    num_cols = min(num_images, 3)
    target_height = 200
    target_width = max(int(target_height * (img.width / img.height))
                       for img in image_paths)

    for i in range(0, num_images, num_cols):
        cols = st.columns(num_cols)
        for idx, col in enumerate(cols):
            if i + idx < num_images:
                with col:
                    img_path = image_paths[i + idx]
                    img_resized_padded = resize_and_pad(
                        img_path, target_height, target_width)
                    st.image(img_resized_padded, use_column_width=True)

                    # Get image embeddings
                    label = st.text_input(
                        f"Label for Image {i + idx + 1}", key=f"label_{i + idx}")

                    if label:  # If user enters something
                        emb = get_query_values(img_path, model, device='cuda')[
                            'det_emb']
                        labels.append(label)
                        embeddings.append(emb)

    if st.button("Saqlash"):
        return labels, embeddings, True
    return labels, embeddings, False


def capture_video(video_path):
    return cv2.VideoCapture(video_path)


def display_frame(frame):
    st.image(frame, channels="RGB")


def save_frame(frame, filename="current_frame.jpg"):
    cv2.imwrite(filename, frame)
    st.success(f"Frame saved as {filename}")
