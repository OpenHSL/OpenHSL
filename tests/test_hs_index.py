import sys
sys.path.insert(1, '../OpenHSL')

import numpy as np 
import openhsl.hs_indexes as indexes
from openhsl.hsi import HSImage
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat


illum = loadmat('./data_rgb_image/illum_6500.mat')
illum_coef = illum['illum_6500']
tmp = loadmat('./data_rgb_image/xyzbar.mat')
xyzbar = tmp['xyzbar']

#загрузить куб для тестов
hsi_cube = loadmat('./data_plazma23/sera3_c3_p3_4x.mat')
hsi_cube = hsi_cube['image']
#print('hsi_cube.shape: ', hsi_cube.shape)

#создать первый объект HSImage без длин волн
hsi_image = HSImage(hsi_cube, None)

#подгрузили длины волн
w_data = np.loadtxt("./420_978.txt", delimiter='\t', dtype=np.int16)

#создать второй объект HSImage с длинами волн
hsi_image2 = HSImage(hsi_cube, list(w_data))


def test_ndvi_mask():
    index_mask_HSI1, index_mask_IMG1 = indexes.ndvi_mask(hsi_cube, w_data)
    index_mask_HSI2, index_mask_IMG2 = indexes.ndvi_mask(hsi_image, w_data)
    index_mask_HSI3, index_mask_IMG3 = indexes.ndvi_mask(hsi_image2, None)
    index_mask_HSI4, index_mask_IMG4 = indexes.ndvi_mask(hsi_image2, w_data)
    

    assert np.allclose(index_mask_HSI1.data, index_mask_HSI2.data) 
    assert np.allclose(index_mask_HSI3.data, index_mask_HSI4.data)
    assert np.allclose(index_mask_HSI3.data, index_mask_HSI2.data)

    assert np.allclose(index_mask_IMG1, index_mask_IMG2) 
    assert np.allclose(index_mask_IMG3, index_mask_IMG4)
    assert np.allclose(index_mask_IMG3, index_mask_IMG2)


def test_dvi_mask():
    index_mask_HSI1, index_mask_IMG1 = indexes.dvi_mask(hsi_cube, w_data)
    index_mask_HSI2, index_mask_IMG2 = indexes.dvi_mask(hsi_image, w_data)
    index_mask_HSI3, index_mask_IMG3 = indexes.dvi_mask(hsi_image2, None)
    index_mask_HSI4, index_mask_IMG4 = indexes.dvi_mask(hsi_image2, w_data)
  

    assert np.allclose(index_mask_HSI1.data, index_mask_HSI2.data) 
    assert np.allclose(index_mask_HSI3.data, index_mask_HSI4.data)
    assert np.allclose(index_mask_HSI3.data, index_mask_HSI2.data)

    assert np.allclose(index_mask_IMG1, index_mask_IMG2) 
    assert np.allclose(index_mask_IMG3, index_mask_IMG4)
    assert np.allclose(index_mask_IMG3, index_mask_IMG2)

    
def test_sr_mask():
    index_mask_HSI1, index_mask_IMG1 = indexes.sr_mask(hsi_cube, w_data)
    index_mask_HSI2, index_mask_IMG2 = indexes.sr_mask(hsi_image, w_data)
    index_mask_HSI3, index_mask_IMG3 = indexes.sr_mask(hsi_image2, None)
    index_mask_HSI4, index_mask_IMG4 = indexes.sr_mask(hsi_image2, w_data)
   

    assert np.allclose(index_mask_HSI1.data, index_mask_HSI2.data) 
    assert np.allclose(index_mask_HSI3.data, index_mask_HSI4.data)
    assert np.allclose(index_mask_HSI3.data, index_mask_HSI2.data)

    assert np.allclose(index_mask_IMG1, index_mask_IMG2) 
    assert np.allclose(index_mask_IMG3, index_mask_IMG4)
    assert np.allclose(index_mask_IMG3, index_mask_IMG2)


def test_osavi_mask():
    index_mask_HSI1, index_mask_IMG1 = indexes.osavi_mask(hsi_cube, w_data)
    index_mask_HSI2, index_mask_IMG2 = indexes.osavi_mask(hsi_image, w_data)
    index_mask_HSI3, index_mask_IMG3 = indexes.osavi_mask(hsi_image2, None)
    index_mask_HSI4, index_mask_IMG4 = indexes.osavi_mask(hsi_image2, w_data)


    assert np.allclose(index_mask_HSI1.data, index_mask_HSI2.data) 
    assert np.allclose(index_mask_HSI3.data, index_mask_HSI4.data)
    assert np.allclose(index_mask_HSI3.data, index_mask_HSI2.data)


    assert np.allclose(index_mask_IMG1, index_mask_IMG2) 
    assert np.allclose(index_mask_IMG3, index_mask_IMG4)
    assert np.allclose(index_mask_IMG3, index_mask_IMG2)


def test_wdrvi_mask():
    index_mask_HSI1, index_mask_IMG1 = indexes.wdrvi_mask(hsi_cube, w_data)
    index_mask_HSI2, index_mask_IMG2 = indexes.wdrvi_mask(hsi_image, w_data)
    index_mask_HSI3, index_mask_IMG3 = indexes.wdrvi_mask(hsi_image2, None)
    index_mask_HSI4, index_mask_IMG4 = indexes.wdrvi_mask(hsi_image2, w_data)



    assert np.allclose(index_mask_HSI1.data, index_mask_HSI2.data) 
    assert np.allclose(index_mask_HSI3.data, index_mask_HSI4.data)
    assert np.allclose(index_mask_HSI3.data, index_mask_HSI2.data)


    assert np.allclose(index_mask_IMG1, index_mask_IMG2) 
    assert np.allclose(index_mask_IMG3, index_mask_IMG4)
    assert np.allclose(index_mask_IMG3, index_mask_IMG2)


def test_wdrvi_mask():
    index_mask_HSI1, index_mask_IMG1 = indexes.wdrvi_mask(hsi_cube, w_data)
    index_mask_HSI2, index_mask_IMG2 = indexes.wdrvi_mask(hsi_image, w_data)
    index_mask_HSI3, index_mask_IMG3 = indexes.wdrvi_mask(hsi_image2, None)
    index_mask_HSI4, index_mask_IMG4 = indexes.wdrvi_mask(hsi_image2, w_data)
   

 
    assert np.allclose(index_mask_HSI1.data, index_mask_HSI2.data)
    assert np.allclose(index_mask_HSI3.data, index_mask_HSI4.data)
    assert np.allclose(index_mask_HSI3.data, index_mask_HSI2.data)

    assert np.allclose(index_mask_IMG1, index_mask_IMG2) 
    assert np.allclose(index_mask_IMG3, index_mask_IMG4)
    assert np.allclose(index_mask_IMG3, index_mask_IMG2)

def test_mtvi2_mask():
    index_mask_HSI1, index_mask_IMG1 = indexes.mtvi2_mask(hsi_cube, w_data)
    index_mask_HSI2, index_mask_IMG2 = indexes.mtvi2_mask(hsi_image, w_data)
    index_mask_HSI3, index_mask_IMG3 = indexes.mtvi2_mask(hsi_image2, None)
    index_mask_HSI4, index_mask_IMG4 = indexes.mtvi2_mask(hsi_image2, w_data)


    assert np.allclose(index_mask_HSI1.data, index_mask_HSI2.data)
    assert np.allclose(index_mask_HSI3.data, index_mask_HSI4.data)
    assert np.allclose(index_mask_HSI3.data, index_mask_HSI2.data)

    assert np.allclose(index_mask_IMG1, index_mask_IMG2)
    assert np.allclose(index_mask_IMG3, index_mask_IMG4)
    assert np.allclose(index_mask_IMG3, index_mask_IMG2)


def test_simple_hsi_to_rgb():
    index_mask_IMG1 = indexes.simple_hsi_to_rgb(hsi_cube, w_data)
    index_mask_IMG2 = indexes.simple_hsi_to_rgb(hsi_image, w_data)
    index_mask_IMG3 = indexes.simple_hsi_to_rgb(hsi_image2, None)
    index_mask_IMG4 = indexes.simple_hsi_to_rgb(hsi_image2, w_data)

  
    assert np.allclose(index_mask_IMG1, index_mask_IMG2)
    assert np.allclose(index_mask_IMG3, index_mask_IMG4)
    assert np.allclose(index_mask_IMG3, index_mask_IMG2)


def test_hsi_to_rgb():
    index_mask_IMG1 = indexes.hsi_to_rgb(hsi_cube, w_data, illum_coef, xyzbar)
    index_mask_IMG2 = indexes.hsi_to_rgb(hsi_image, w_data, illum_coef, xyzbar)
    index_mask_IMG3 = indexes.hsi_to_rgb(hsi_image2, None, illum_coef, xyzbar)
    index_mask_IMG4 = indexes.hsi_to_rgb(hsi_image2, w_data, illum_coef, xyzbar)


    assert np.allclose(index_mask_IMG1, index_mask_IMG2)
    assert np.allclose(index_mask_IMG3, index_mask_IMG4)
    assert np.allclose(index_mask_IMG3, index_mask_IMG2)

  