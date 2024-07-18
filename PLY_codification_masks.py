import cv2
import os
import json
import numpy as np

mask_directory = 'Path to the masks generated after running the "extract_masks.py" script'
original_json_path = 'Path to the original JSON file from the Chula-ParasiteEgg-11 Dataset, publicly available'
output_json_path = 'Path to the new JSON file that will be generated adding segmentation information'

# Open original JSON
with open(original_json_path, "r") as json_file:
    data = json.load(json_file)

for annotation in data["annotations"]:
    image_info = next(image for image in data["images"] if
                      image["id"] == annotation["image_id"])
    
    mask_filename = image_info["file_name"].replace(".jpg","_mask.jpg")

    # Load mask
    mask_image = cv2.imread(os.path.join(mask_directory,mask_filename), cv2.IMREAD_GRAYSCALE)

    height, width = mask_image.shape[:2]
    bbox = annotation["bbox"]
    x = int(bbox[0])
    y = int(bbox[1])
    bb_width = int(bbox[2])
    bb_height = int(bbox[3])
    protected_region = mask_image[y:y+bb_height, x:x+bb_width]
    my_mask = np.zeros((height, width), dtype=np.uint8)
    my_mask[y:y+bb_height, x:x+bb_width] = protected_region

    # Canny Edge Detector
    t_lower = 0 # Lower Threshold 
    t_upper = 255 # Upper threshold 
    aperture_size = 5
    L2Gradient = True
    edge_detected_image = cv2.Canny(my_mask, t_lower, t_upper, apertureSize=aperture_size, L2gradient=L2Gradient) 

    # Find contours
    contours,_ = cv2.findContours(edge_detected_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Convert contours to Polygon (PLY) format
    segmentation = []
    for contour in contours:
        if len(contour) > 4:
            segmentation.append(contour.flatten().tolist())
        

    if segmentation:        
        annotation["iscrowd"] = 0
        annotation["segmentation"] = segmentation
    else:
        annotation["iscrowd"] = 0
        annotation["segmentation"] = [[]]

# Guardamos el JSON
with open("Path to new JSON adding segmentation information", 'w') as json_file:
    json.dump(data, json_file, indent=4)
    
