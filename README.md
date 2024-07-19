# Segmented-Chula-ParasiteEgg-11-CODE
This repository contains the code used to manipulate the original Chula-ParasiteEgg-11 in order to generate the segmentation masks of the images as well as the JSON files with the information.

## Previous steps
It is mandatory to have the original dataset (available at https://ieee-dataport.org/competitions/parasitic-egg-detection-and-classification-microscopic-images) downloaded. Furthermore, to obtain the segmentation masks it is used the SAM, which can be done easily trough executing the following commmand:
```bash
pip install segment-anything
```
Of course, it is necessary to download the model itself. Download a checkpoint [here](https://github.com/facebookresearch/segment-anything#model-checkpoints). In the present code, it was chosen to use the *vit_h* one.

## Scripts execution order
Note that the original dataset is distributed in two sets (train/validation with 11000 images and test with 2200 images) with two JSON files each one. In case the user chooses not to mix these sets, it will be necessary to execute the code two times, one per set of images (with its corresponding JSON file). The execution order of the code would be as follows:

- ***extract_masks.py***:
  
  Modify the paths and execute it to generate the masks.
  
- ***PLY_codification_masks.py***:
  
  Modify the paths and execute it to generate the new JSON file with the segmentation masks information.
