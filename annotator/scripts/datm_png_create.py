# This script uses the prepped src files and outputs PNG files into the directory of choice having two classes
# defect_0 and defect_1 according to the threshold settings.

import cv2
import numpy as np
import math
import shutil
import os
from tqdm import tqdm
import random
import time

# Make nondefect segment generation repeatable
random.seed(2323)

# SRC folders etc.
SRC_FOLDER = "C:\\Data\\ReachU-3DGIS\\defect-detection-ext-paper\\20190414_083725_LD5_PREPPED_SRC" # No trailing slash
PNG_FOLDER = "C:\\Data\\ReachU-3DGIS\\defect-detection-ext-paper\\20190414_083725_LD5_PNG"
THR_IMAGE = 1.0 # Only consider pure segments without any mask
THR_DEFECT = 0.05 # if NN% of segment pixels are marked as defect, use the segment as defect
SEG_WH = (224, 224) # Segment size

# This returns the original masks
def unpack_masks(orig_mask):
    img_mask = np.zeros(orig_mask.shape, np.uint8)
    def_mask = np.zeros(orig_mask.shape, np.uint8)

    # Perhaps this can be done more efficiently, but it's still quite fast, so no problem
    img_mask[np.where((orig_mask == [255, 0, 0]).all(axis=2) |
                      (orig_mask == [0,255,0]).all(axis=2))] = [255, 255, 255]
    def_mask[np.where((orig_mask == [255, 0, 0]).all(axis=2))] = [255, 255, 255]

    # Masks must be grayscale as well
    img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
    def_mask = cv2.cvtColor(def_mask, cv2.COLOR_BGR2GRAY)

    return (img_mask, def_mask)

# Percent of nonblack pixels in given rectangle
def seg_get_nonblack_pixel_percentage(mask, ctuple):
    x, y, w, h = ctuple
    seg = mask[y:y + h, x:x + w]
    return cv2.countNonZero(seg) / (w * h)

def seg_preparse_image(mask, segwh, thr):

    seg_width, seg_height = segwh
    h, w = mask.shape
    if seg_width > w or seg_height > h:
        print("Segment size is larger than the image: cannot proceed")
        return

    # Number of segments per line and row
    numSegsX = int(math.floor(w / seg_width))
    numSegsY = int(math.floor(h / seg_height))

    # Segment list
    seg_list = []

    # Segmentation should be done in four steps:
    # 1. Top to bottom, left to right
    # 2. Rightmost boundary: take all segments from the right-end pixel along the vertical (top to bottom)
    # 3. Bottommost boundary: take all segments from the bottom-end pixel along the horizontal (left to right)
    # 4. Final segment: the last segment located in the bottom right.
    # Unless masked, these regions will be overrepresented slightly.

    # Step 1
    for sy in range(numSegsY):
        for sx in range(numSegsX):
            px, py, pw, ph = (sx * seg_width, sy * seg_height, seg_width, seg_height)
            if seg_get_nonblack_pixel_percentage(mask, (px, py, pw, ph)) >= thr:
                seg_list.append((px, py, pw, ph))

    # Step 2
    for sy in range(numSegsY):
        px, py, pw, ph = (w - seg_width, sy * seg_height, seg_width, seg_height)
        if seg_get_nonblack_pixel_percentage(mask, (px, py, pw, ph)) >= thr:
            seg_list.append((px, py, pw, ph))

    # Step 3
    for sx in range(numSegsX):
        px, py, pw, ph = (sx * seg_width, h - seg_height, seg_width, seg_height)
        if seg_get_nonblack_pixel_percentage(mask, (px, py, pw, ph)) >= thr:
            seg_list.append((px, py, pw, ph))

    # Step 4
    px, py, pw, ph = (w - seg_width, h - seg_height, seg_width, seg_height)
    if seg_get_nonblack_pixel_percentage(mask, (px, py, pw, ph)) >= thr:
        seg_list.append((px, py, pw, ph))

    return seg_list

##### SCRIPT BEGINS HERE #####

# Create the new dir as needed
if os.path.exists(PNG_FOLDER):
    shutil.rmtree(PNG_FOLDER)
    os.makedirs(PNG_FOLDER)
    os.mkdir(PNG_FOLDER + os.sep + "defect_0")
    os.mkdir(PNG_FOLDER + os.sep + "defect_1")
else:
    os.makedirs(PNG_FOLDER)
    os.mkdir(PNG_FOLDER + os.sep + "defect_0")
    os.mkdir(PNG_FOLDER + os.sep + "defect_1")

# Get all files
all_files = os.listdir(SRC_FOLDER)
for k in tqdm(range(len(all_files))):
    now_file = all_files[k]
    now_file_ne = now_file.split(".")[0]
    # We process images according to .img.npy and load the masks automatically
    if ".img.npy" in now_file:
        # Load the files
        img = np.load(SRC_FOLDER + os.sep + now_file_ne + ".img.npy")
        msk = np.load(SRC_FOLDER + os.sep + now_file_ne + ".masks.npy")

        # Unpack masks: time consuming operation. Can we make this faster?
        im_msk, df_msk = unpack_masks(msk)

        # Now we compile the list of all segments
        segs = seg_preparse_image(im_msk, SEG_WH, THR_IMAGE)

        # We now need to determine which segments contain defects and which do not
        # So we create more lists
        segs_no_def = []
        segs_yes_def = []
        for seg in segs:
            if seg_get_nonblack_pixel_percentage(df_msk, seg) >= THR_DEFECT:
                segs_yes_def.append(seg)
            else:
                segs_no_def.append(seg)

        # Now that we know where the defects are, we can write them out to the folders
        num_defects = len(segs_yes_def)

        # Store defects
        for k in range(num_defects):
            px, py, pw, ph = segs_yes_def[k]
            now_seg = img[py:py+pw, px:px+pw]
            cv2.imwrite(PNG_FOLDER + os.sep + "defect_1" + os.sep
                        + now_file_ne + ("_%04d" % k) + ".png", now_seg)

        # Store random non-defects
        for k in range(num_defects):
            i = random.choice(range(len(segs_no_def)))
            px, py, pw, ph = segs_no_def[i]
            now_seg = img[py:py+pw, px:px+pw]
            cv2.imwrite(PNG_FOLDER + os.sep + "defect_0" + os.sep
                        + now_file_ne + ("_%04d" % k) + ".png", now_seg)