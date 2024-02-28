# Prepare source files for segment extraction
# NB! LOTS of disk space required
import cv2
import numpy as np
import os
from tqdm import tqdm
import shutil
import time

# Some file naming conventions
ORIG_IMG = ".jpg"
CUT_MASK_V1 = ".cut.mask.png"
CUT_MASK_V2 = ".cut.mask_v2.png"
DEFECT_MASK = ".defect.mask.png"

# Saving
SAVE_IMG = ".img.npy"
COMB_MASK = ".masks.npy"

# Path to where the images with .defect.mask.png and .cut.mask_v2.png are stored
# NB! DO NOT, I repeat, DO NOT choose POST_SRC_FOLDER the same as PRE_SRC_FOLDER
# This is because it is CLEARED of ALL FILES on every run
# NEVER choose POST_SRC_FOLDER as folder that already contains some data, otherwise you will lose it!
PRE_SRC_FOLDER = "C:\\Data\\ReachU-3DGIS\\defect-detection-ext-paper\\20190414_083725_LD5"
POST_SRC_FOLDER = "C:\\Data\\ReachU-3DGIS\\defect-detection-ext-paper\\20190414_083725_LD5_PREPPED_SRC" # NEW FOLDER!

# Create the new dir as needed
if os.path.exists(POST_SRC_FOLDER):
    shutil.rmtree(POST_SRC_FOLDER)
    os.makedirs(POST_SRC_FOLDER)
else:
    os.makedirs(POST_SRC_FOLDER)

# Get list of all images from the pre_src
all_presrc_files = os.listdir(PRE_SRC_FOLDER)

# Go through all the files checking if we have detected a .defect.mask.png file
# If detected, prep the file for saving along with the original image. For speed, files are saved as NPY arrays

# Format for masks: black = ignore, red = defect, green = no defect (according to supplied annotation)

# Let's go.
for n in tqdm(range(len(all_presrc_files))):
    # File name
    myfile = all_presrc_files[n]

    # Found it
    if DEFECT_MASK in myfile:
        myfile_no_ext = myfile.split(".")[0]

        # Load the original image
        img = cv2.imread(PRE_SRC_FOLDER + os.sep + myfile_no_ext + ORIG_IMG)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Check which mask to load
        mask = None
        if os.path.isfile(PRE_SRC_FOLDER + os.sep + myfile_no_ext + CUT_MASK_V2):
            mask = cv2.imread(PRE_SRC_FOLDER + os.sep + myfile_no_ext + CUT_MASK_V2)
        else:
            mask = cv2.imread(PRE_SRC_FOLDER + os.sep + myfile_no_ext + CUT_MASK_V1)

        # Load the defect mask
        dmask = cv2.imread(PRE_SRC_FOLDER + os.sep + myfile)

        # First, clip defect mask to actual mask
        dmask[np.where((mask == [0, 0, 0]).all(axis=2))] = [0, 0, 0]

        # Now, create a new mask with colors
        nmask = np.zeros(dmask.shape, np.uint8)

        # Fill original mask first
        nmask[np.where((mask == [255, 255, 255]).all(axis=2))] = [0, 255, 0]
        nmask[np.where((dmask == [255, 255, 255]).all(axis=2))] = [255, 0, 0]

        # Save NPY arrays
        np.save(POST_SRC_FOLDER + os.sep + myfile_no_ext + SAVE_IMG, img)
        np.save(POST_SRC_FOLDER + os.sep + myfile_no_ext + COMB_MASK, nmask)