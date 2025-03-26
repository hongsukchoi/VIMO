import sys
import os
import os.path as osp
sys.path.insert(0, os.path.dirname(__file__) + '/..')
import json
import torch
import argparse
import numpy as np
import cv2
from pathlib import Path
from glob import glob

import pickle
from vimo.models import get_hmr_vimo


# Convert field of view (in degrees) to focal length
def fov2focal(fov_deg, size):
    """Convert field of view in degrees to focal length in pixels.
    
    Args:
        fov_deg: Field of view in degrees
        size: Image dimension (width or height) in pixels
    Returns:
        focal: Focal length in pixels
    """
    fov_rad = np.deg2rad(fov_deg)
    focal = size / (2 * np.tan(fov_rad / 2))
    return focal

parser = argparse.ArgumentParser()
parser.add_argument('--img-dir', type=str, default='../human_in_world/demo_data/input_images/IMG_7415/cam01', help='input video')
parser.add_argument('--mask-dir', type=str, default='../human_in_world/demo_data/input_masks/IMG_7415/cam01', help='input video')
parser.add_argument('--out-dir', type=str, default='../human_in_world/demo_data/input_3d_mesh_vimo/IMG_7415/cam01', help='output directory')
args = parser.parse_args()


Path(args.out_dir).mkdir(parents=True, exist_ok=True)

# Retrieve the person ids from the meta data in sam2 output using bbox_dir
# Load the meta data
with open(osp.join(args.mask_dir, 'meta_data.json'), 'r') as f:
    meta_data = json.load(f)
# {"most_persistent_id": 1, "largest_area_id": 1, "highest_confidence_id": 1}
person_id_list = [meta_data["most_persistent_id"], meta_data["largest_area_id"], meta_data["highest_confidence_id"]]
# Do the majority voting across the three person ids
from collections import Counter
person_id_counter = Counter(person_id_list)
majority_voted_person_id = person_id_counter.most_common(1)[0][0]
print(f"Majority voted person id: {majority_voted_person_id} using most persistent, largest area, and highest confidence")
person_ids = [majority_voted_person_id]

# load bounding boxes
bbox_files = sorted(glob(f'{args.mask_dir}/json_data/mask_*.json'))
bbox_list = []
frame_idx_list = []

for bbox_file in bbox_files:
    frame_idx = int(Path(bbox_file).stem.split('_')[-1])
    with open(bbox_file, 'r') as f:
        bbox_data = json.load(f)
        # if value of "labels" key is empty, continue
        if not bbox_data['labels']:
            continue
        else:
            labels = bbox_data['labels']
            # "labels": {"1": {"instance_id": 1, "class_name": "person", "x1": 454, "y1": 399, "x2": 562, "y2": 734, "logit": 0.0}, "2": {"instance_id": 2, "class_name": "person", "x1": 45, "y1": 301, "x2": 205, "y2": 812, "logit": 0.0}}}
            label_keys = sorted(labels.keys())

            # filter label keys by person ids
            selected_label_keys = [x for x in label_keys if labels[x]['instance_id'] in person_ids]
            label_keys = selected_label_keys

            # get boxes
            boxes = np.array([[labels[str(i)]['x1'], labels[str(i)]['y1'], labels[str(i)]['x2'], labels[str(i)]['y2']] for i in label_keys])

            # sanity check; if boxes is empty, continue
            if boxes.sum() == 0:
                continue

            # add to the lists
            bbox_list.append(boxes)
            frame_idx_list.append(frame_idx)


# get img paths
imgfiles = []
for frame_idx in frame_idx_list:
    imgfiles.append(osp.join(args.img_dir, f'{frame_idx:05d}.jpg'))

assert len(imgfiles) == len(bbox_list)

# # TEMP
# imgfiles = imgfiles[:30]
# bbox_list = bbox_list[:30]
# frame_idx_list = frame_idx_list[:30]

# load one image to get camera parameters
img = cv2.imread(imgfiles[0])
fov = 60
# fov to focal
# Convert vertical FOV to focal length using image height
img_focal = None # fov2focal(fov, img.shape[0])
img_center = None # (img.shape[1] / 2, img.shape[2] / 2)

bboxes = np.concatenate(bbox_list, axis=0, dtype=np.float32)
##### Run HPS (here we use tram) #####
print('Estimate HPS ...')
model = get_hmr_vimo(checkpoint='/home/arthur/code2/human_in_world/tram/data/pretrain/vimo_checkpoint.pth.tar')

results = model.inference(imgfiles, bboxes, img_focal=img_focal, img_center=img_center)

# Save the results

# 'global_orient': (1, 3, 3), 'body_pose': (23, 3, 3), 'betas': (10)

pred_rotmat = results['pred_rotmat'] # (num_frames, 24, 3, 3)
pred_shape = results['pred_shape'] # (num_frames, 10)

for f_idx, frame_idx in enumerate(frame_idx_list):
    frame_result_save_path = os.path.join(args.out_dir, f'smpl_params_{frame_idx:05d}.pkl')

    result_dict = {
        majority_voted_person_id: {
            'smpl_params': {
                'global_orient': pred_rotmat[f_idx][0:1, :, :].cpu().numpy(),
                'body_pose': pred_rotmat[f_idx][1:, :, :].cpu().numpy(),
                'betas': pred_shape[f_idx].cpu().numpy(),
            }
        }
    }
    with open(frame_result_save_path, 'wb') as f:
        pickle.dump(result_dict, f)
