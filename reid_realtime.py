from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import ssl
from PIL import Image
import os
import torchvision.transforms.functional as TF
from albumentations.augmentations.geometric import functional as FGeometric
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import cv2
# Torch libs
import torch
# Disable nvfuser for now
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_nvfuser_enabled(False)

# Libs for loading images
# Avoid SSL error
ssl._create_default_https_context = ssl._create_unverified_context

# Libs for visualization


# Convert PIL image to torch tensor
def to_tensor(image):
    arr = np.array(image)
    arr_wrs = window_resize(arr)
    print(arr_wrs.shape)
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


def get_query_values(image_path, model, device):
    image = Image.open(image_path).convert("RGB")
    query_tensor = to_tensor(image)
    query_box = torch.FloatTensor(
        [0, 0, *query_tensor.shape[1:]]).unsqueeze(0).to(device)
    query_targets = [{'boxes': query_box}]
    with torch.no_grad():
        detections = model([query_tensor], query_targets,
                           inference_mode='both')
    query_detect = detections[0]
    # show_detects(query_tensor, query_detect, show_detect_score=True)
    return query_detect, query_tensor


def get_gallery_values(frame, model, device):
    frame_tensor = to_tensor(frame).to(device)

    with torch.no_grad():
        detections_list = model([frame_tensor], inference_mode='det')
    return detections_list[0]


def compute_person_similarity(query_output_list, gallery_output_list):
    for gallery_output_dict in gallery_output_list:
        if 'person_sim' not in gallery_output_dict:
            gallery_output_dict['person_sim'] = []
    for query_detect in query_output_list:
        query_person_emb = query_detect['det_emb']
        for gallery_output_dict in gallery_output_list:
            gallery_person_emb = gallery_output_dict['det_emb']
            person_sim = torch.mm(
                F.normalize(query_person_emb, dim=1),
                F.normalize(gallery_person_emb, dim=1).T
            ).flatten()
            gallery_output_dict['person_sim'].append(person_sim)
    return gallery_output_list


def compute_gfn_scores(query_output_list, gallery_output_list, model):
    for gallery_output_dict in gallery_output_list:
        if 'gfn_sim' not in gallery_output_dict:
            gallery_output_dict['gfn_sim'] = []
    for query_detect in query_output_list:
        query_person_emb = query_detect['det_emb']
        query_scene_emb = query_detect['scene_emb']
        for gallery_output_dict in gallery_output_list:
            gallery_scene_emb = gallery_output_dict['scene_emb']
            with torch.no_grad():
                qg_scene_sim = model.gfn.get_scores(
                    query_person_emb, query_scene_emb, gallery_scene_emb).flatten().item()
            gallery_output_dict['gfn_sim'].append(qg_scene_sim)
    return gallery_output_list


def get_full_score_list(query_idx, gallery_output_list):
    full_score_list = []
    for gallery_output_dict in gallery_output_list:
        person_sim = gallery_output_dict['person_sim'][query_idx]
        full_score_list.extend(person_sim.tolist())
    return full_score_list


query_image_path = 'notebooks/demo_images/ross_query.jpg'
video_path = 'notebooks/demo_images/videoplayback2.mp4'
device = torch.device('cpu')
model = torch.jit.load(
    'torchscript/cuhk_final_convnext-base_e30.torchscript.pt', map_location=device)


query_detect, _ = get_query_values(query_image_path, model, device)

cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps
one_minute_frames = int(fps * 10)

# print(cap.isOpened())
frame_files = []
with tqdm(total=min(one_minute_frames, frame_count), desc="Processing Video Frames") as pbar:
    for frame_idx in range(min(one_minute_frames, frame_count)):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1200, 900))
        if not ret:
            break

        print(frame.shape)
        gallery_detect = get_gallery_values(
            frame, model, device)
        print("query_detect['det_emb'].shape", query_detect['det_emb'].shape)
        person_sim = torch.mm(
            F.normalize(query_detect['det_emb'], dim=1),
            F.normalize(gallery_detect['det_emb'], dim=1).T
        ).flatten()

        detection_boxes = gallery_detect['det_boxes'].cpu().tolist()

        for i, (box, score) in enumerate(zip(detection_boxes, person_sim)):
            x, y, x2, y2 = box
            x, y, x2, y2 = int(x), int(y), int(x2), int(y2)

            print(x, y, x2, y2, frame.shape)

            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 4)
            cv2.putText(frame, f'{person_sim[i].item() * 100:.2f}%', (int(
                x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255), 2)

        out.write(frame)
        frame = cv2.resize(frame, (640, 480))
        cv2.imshow('Frame', frame)
        # make quit by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        pbar.update(1)

cap.release()
out.release()
cv2.destroyAllWindows()
