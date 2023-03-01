from hs_builder import HSBuilder
from matplotlib import pyplot as plt
from hsi import HSImage

test_wavelengths = [i for i in range(400, 650)]


def test_hs_builder_imgs_rail():
    hsb = HSBuilder(path_to_data='./test_data/builder/imgs',
                    data_type='images')
    hsb.build(roi=True)
    hsi = hsb.get_hsi()

    # Проверка размерности ГСИ
    assert (250, 900, 250) == hsi.data.shape

    # Генерируем набор длин волн для теста
    hsi.wavelengths = test_wavelengths

    # Проверка возможности сохранения в h5
    hsi.save_to_h5(path_to_file='./out/microscope.h5',
                   h5_key='image')

    hsi = HSImage(None, None)

    # Проверка чтения из h5
    hsi.load_from_h5(path_to_file='./out/microscope.h5',
                     h5_key='image')

    assert test_wavelengths == hsi.wavelengths

    plt.imshow(hsi.data[:, :, 100], cmap='gray')
    plt.show()
# ----------------------------------------------------------------------------------------------------------------------

"""
def test_hs_builder_video_rotary():
    # Сборка из штатива
    hsb = HSBuilder(path_to_data='./test_data/builder/video/rec_2022-06-06-12-24-02.avi',
                    data_type='video')
    hsb.build()
    hsi = hsb.get_hsi()

    # Проверка размерности ГСИ
    assert (1001, 2048, 274) == hsi.data.shape

    # Генерируем набор длин волн для теста
    hsi.wavelengths = test_wavelengths

    # Проверка возможности сохранения в mat
    hsi.save_to_mat(path_to_file='./out/tripod.mat',
                    mat_key='image')

    hsi = HSImage(None, None)

    # Проверка чтения из mat
    hsi.load_from_mat(path_to_file='./out/tripod.mat',
                      mat_key='image')

    assert test_wavelengths == hsi.wavelengths

    plt.imshow(hsi.data[:, :, 100], cmap='gray')
    plt.show()
# ----------------------------------------------------------------------------------------------------------------------


def test_hs_builder_video_uav():
    # Сборка из данных коптера
    hsb = HSBuilder(path_to_data='./test_data/builder/copter',
                    path_to_metadata='./test_data/builder/copter/gps_2021-03-30.csv',
                    data_type='video')
    hsb.build(principal_slices=True)
    hsi = hsb.get_hsi()

    # Проверка размерности ГСИ
    assert (1724, 1080, 40) == hsi.data.shape

    # Проверка возможности сохранения в npy
    hsi.save_to_npy(path_to_file='./out/uav.npy')

    hsi = HSImage(None, None)

    # Проверка чтения из npy
    hsi.load_from_npy(path_to_file='./out/uav.npy')

    plt.imshow(hsi.data[:, :, 20], cmap='gray')
    plt.show()
# ----------------------------------------------------------------------------------------------------------------------
"""
