import cv2
# Torch libs
import torch
## Disable nvfuser for now
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_nvfuser_enabled(False)
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from albumentations.augmentations.geometric import functional as FGeometric
import torchvision.transforms.functional as TF

# Libs for loading images
import os
from PIL import Image
import ssl
## Avoid SSL error
ssl._create_default_https_context = ssl._create_unverified_context

# Libs for visualization
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# Convert PIL image to torch tensor
def to_tensor(image):
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

def get_query_values(image_path, model, device):
    image = Image.open(image_path).convert("RGB")
    query_tensor = to_tensor(image)
    query_box = torch.FloatTensor([0, 0, *query_tensor.shape[1:]]).unsqueeze(0).to(device)
    query_targets = [{'boxes': query_box}]
    with torch.no_grad():
        detections = model([query_tensor], query_targets, inference_mode='both')
    query_detect = detections[0]
    # show_detects(query_tensor, query_detect, show_detect_score=True)
    return query_detect, query_tensor

def get_gallery_values(gallery_image_list, model, device):
    gallery_output_list = []
    gallery_tensor_list = [to_tensor(image).to(device) for image in gallery_image_list]
    with torch.no_grad():
        detections = model(gallery_tensor_list, inference_mode='det')
    for tensor, detect in zip(gallery_tensor_list, detections):
        # show_detects(tensor, detect, show_detect_score=True)
        gallery_output_list.append(detect)
    return gallery_output_list, gallery_tensor_list

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
                qg_scene_sim = model.gfn.get_scores(query_person_emb, query_scene_emb, gallery_scene_emb).flatten().item()
            gallery_output_dict['gfn_sim'].append(qg_scene_sim)
    return gallery_output_list

def get_full_score_list(query_idx, gallery_output_list):
    full_score_list = []
    for gallery_output_dict in gallery_output_list:
        person_sim = gallery_output_dict['person_sim'][query_idx]
        full_score_list.extend(person_sim.tolist())
    return full_score_list


query_image_path = 'ross_query.jpg'
video_path = 'videoplayback2.mp4'
device = torch.device('cpu')
model = torch.jit.load('cuhk_final_convnext-base_e30.torchscript.pt', map_location=device)


output_frames_dir = 'frames'
os.makedirs(output_frames_dir, exist_ok=True)

query_detect, query_tensor = get_query_values(query_image_path, model, device)

cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps
one_minute_frames = int(fps * 10)

#print(cap.isOpened())
frame_files = []
with tqdm(total=min(one_minute_frames, frame_count), desc="Processing Video Frames") as pbar:
    for frame_idx in range(min(one_minute_frames, frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        gallery_image_list = [frame]
        #print(frame.shape)
        gallery_output_list, gallery_tensor_list = get_gallery_values(gallery_image_list, model, device)
        gallery_output_list = compute_person_similarity([query_detect], gallery_output_list)
        gallery_output_list = compute_gfn_scores([query_detect], gallery_output_list, model)
        #full_score_list = get_full_score_list(0, gallery_output_list)
        #print(gallery_output_list)
    
        for gallery_output_dict in gallery_output_list:
            detection_boxes = gallery_output_dict['det_boxes'].cpu().tolist()
            person_sim = gallery_output_dict['person_sim']    
            for i, (box, score) in enumerate(zip(detection_boxes, person_sim)): 
                x, y, x2, y2 = box
                x, y, x2, y2 = int(x), int(y), int(x2), int(y2)
                #y = frame.shape[0] - y
                #y2 = frame.shape[0] - y2
                w, h = x2 - x, y2 - y
                print(x2, y2)
                #ax.add_patch(Rectangle((x, y), w, h, edgecolor='green', lw=4, fill=False, alpha=0.8))
                
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 4)
                cv2.putText(frame, f'{person_sim[i].item() * 100:.2f}%', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255), 2)
        
        # cv2.imshow('frame', frame)
           
        #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #plt.imshow(frame_rgb)
        #plt.axis('off')  # Turn off axis
        #frame_filename = os.path.join(output_frames_dir, f'frame_{frame_idx:04d}.png')
        #plt.savefig(frame_filename, bbox_inches='tight', pad_inches=0)
        #frame_files.append(frame_filename)
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