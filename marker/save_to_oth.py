from matplotlib import pyplot as plt
import matplotlib.cm as cm
from PIL import Image as im

from hsi import HSImage
from hs_mask import HSMask 

import numpy as np

'''
bg_image_file_path = "C:/Users/retuo/Downloads/coffe/PaviaU.mat" # PaviaU   PaviaU_gt
key_answer = "paviaU"

hsi = HSImage()
hsi.load_from_mat(path_to_file=bg_image_file_path, mat_key= key_answer)

print(np.array(hsi).shape[0])

#data_image_file_path_2 = "C:/Users/retuo/Downloads/coffe/PaviaU_test.h5" # /three_coffee_piles_2.mat
#hsi.save_to_h5(path_to_file=data_image_file_path_2, h5_key= key_answer)

#data_image_file_path_2 = "C:/Users/retuo/Downloads/coffe/PaviaU_test.npy" # /three_coffee_piles_2.mat
#hsi.save_to_npy(path_to_file=data_image_file_path_2)
'''


bg_image_file_path = "C:/Users/retuo/Downloads/coffe/PaviaU_gt.mat" # PaviaU   PaviaU_gt
key_answer = "paviaU_gt"

hsi = HSMask()
hsi.load_mask(path_to_file=bg_image_file_path, mat_key= key_answer, h5_key= key_answer)

layer = 7

l = np.array(hsi).shape[0]
h = np.array(hsi).shape[1]
w = np.array(hsi).shape[2]
plt.imsave('filename.png', np.array(hsi[layer]).reshape(h,w), cmap=cm.magma) # viridis