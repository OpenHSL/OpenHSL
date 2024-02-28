import cv2
import numpy as np
import os
from tqdm import tqdm
import shutil
import math
import random

# Make nondefect segment generation repeatable
random.seed(2323)

# Path to where the images with .defect.mask.png and .cut.mask_v2.png are stored
# NB! DO NOT, I repeat, DO NOT choose POST_SRC_FOLDER the same as PRE_SRC_FOLDER
# This is because it is CLEARED of ALL FILES on every run
# NEVER choose POST_SRC_FOLDER as folder that already contains some data, otherwise you will lose it!
PRE_SRC_FOLDER = "C:\\Data\\ReachU-3DGIS\\defect-detection-ext-paper\\20190414_083725_LD5"
PNG_FOLDER = "C:\\Data\\ReachU-3DGIS\\defect-detection-ext-paper\\20190414_083725_LD5_PNG" # NEW FOLDER!

# Some file naming conventions
ORIG_IMG = ".jpg"
CUT_MASK_V1 = ".cut.mask.png"
CUT_MASK_V2 = ".cut.mask_v2.png"
DEFECT_MASK = ".defect.mask.png"

THR_IMAGE = 1.0 # Only consider pure segments without any mask
THR_DEFECT = 0.05 # if NN% of segment pixels are marked as defect, use the segment as defect
SEG_WH = (224, 224) # Segment size

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

##### SCRIPT BEGINS HERE

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

# Files for prescreening
all_prescr_files = os.listdir(PRE_SRC_FOLDER)

files_with_defects_noext = []
for fil in all_prescr_files:
    if DEFECT_MASK in fil:
        files_with_defects_noext.append(fil.split(".")[0])

# Now we start processing
for n in tqdm(range(len(files_with_defects_noext))):
    # File name
    myfile = files_with_defects_noext[n]

    # Load the original image
    img = cv2.imread(PRE_SRC_FOLDER + os.sep + myfile + ORIG_IMG)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Check which mask to load
    mask = None
    if os.path.isfile(PRE_SRC_FOLDER + os.sep + myfile + CUT_MASK_V2):
        mask = cv2.imread(PRE_SRC_FOLDER + os.sep + myfile + CUT_MASK_V2)
    else:
        mask = cv2.imread(PRE_SRC_FOLDER + os.sep + myfile + CUT_MASK_V1)

    # Load the defect mask
    dmask = cv2.imread(PRE_SRC_FOLDER + os.sep + myfile + DEFECT_MASK)

    # First, clip defect mask to actual mask
    dmask[np.where((mask == [0, 0, 0]).all(axis=2))] = [0, 0, 0]

    # Make masks greyscale
    img_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    def_mask = cv2.cvtColor(dmask, cv2.COLOR_BGR2GRAY)

    # Now we compile the list of all segments
    segs = seg_preparse_image(img_mask, SEG_WH, THR_IMAGE)

    # We now need to determine which segments contain defects and which do not
    # So we create more lists
    segs_no_def = []
    segs_yes_def = []
    for seg in segs:
        if seg_get_nonblack_pixel_percentage(def_mask, seg) >= THR_DEFECT:
            segs_yes_def.append(seg)
        else:
            segs_no_def.append(seg)

    # Now that we know where the defects are, we can write them out to the folders
    num_defects = len(segs_yes_def)

    # Store defects
    for k in range(num_defects):
        px, py, pw, ph = segs_yes_def[k]
        now_seg = img[py:py + pw, px:px + pw]
        cv2.imwrite(PNG_FOLDER + os.sep + "defect_1" + os.sep
                    + myfile + ("_%04d" % k) + ".png", now_seg)

    # Store random non-defects
    for k in range(num_defects):
        i = random.choice(range(len(segs_no_def)))
        px, py, pw, ph = segs_no_def[i]
        now_seg = img[py:py + pw, px:px + pw]
        cv2.imwrite(PNG_FOLDER + os.sep + "defect_0" + os.sep
                    + myfile + ("_%04d" % k) + ".png", now_seg)