import cv2
import json
import os
import shutil
import numpy as np
from segment_anything import SamPredictor
import torch
from segment_anything import sam_model_registry

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = 'Path to your checkpoint model'
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)


dir_path = 'Path to the original jpg images from the Chula-ParasiteEgg-11 Dataset, publicly available'
json_path = 'Path to the original JSON file from the Chula-ParasiteEgg-11 Dataset, publicly available'
output_dir_path = 'Path to the output dir where the segmentation masks will be stored'

if os.path.exists(output_dir_path):
    shutil.rmtree(output_dir_path)
else:
    os.makedirs(output_dir_path)

# Open original JSON
with open(json_path, "r") as json_file:
    data = json.load(json_file)


# Initiate the mask predictor
mask_predictor = SamPredictor(sam)

file_list = os.listdir(dir_path)

for image in file_list:
    frame = cv2.imread(os.path.join(dir_path,image))

    if frame is None:
        print(f"Can't read the image: {os.path.join(dir_path,image)}")
        continue

    for frame_data in data["images"]:
        if frame_data["file_name"] == image:
            id = frame_data["id"]
            
            bounding_boxes = []
            for ann_data in data["annotations"]:
                ann_id = ann_data["image_id"]
                if ann_id == id:
                    # Bbox:
                    bbox = ann_data["bbox"]
                    # Mask:
                    bbox[2] = bbox [0] + bbox [2]
                    bbox[3] = bbox [1] + bbox [3]
                    bounding_boxes.append(bbox)

            mask_predictor.set_image(frame)
            input_boxes= torch.tensor(bounding_boxes, device=mask_predictor.device)
            transformed_boxes = mask_predictor.transform.apply_boxes_torch(input_boxes, frame.shape[:2])

            # Predict mask
            masks, _, _ = mask_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
        
            all_masks = [] # List with all masks (in case there are two or more for a single image)

            for mask_index in range(len(masks)):
                current_mask = masks[mask_index].cpu().numpy()
                binary_mask = current_mask.astype(int)
                binary_mask = (binary_mask*255).astype(np.uint8)
                binary_mask = np.squeeze(binary_mask)
                all_masks.append(binary_mask)
                
            # Combine masks and generate the final one
            combined_mask = np.zeros_like(all_masks[0])
            for mask in all_masks:
                combined_mask = np.bitwise_or(combined_mask, mask)

            mask_name = image.rpartition(".jpg")[0]+f"_mask.jpg"


            # Save mask
            output_path_full_mask = os.path.join(output_dir_path, mask_name)
            cv2.imwrite(output_path_full_mask,combined_mask)

print(f"Masks generated and stored in: {output_dir_path}")