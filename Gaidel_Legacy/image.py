import cv2
from sklearn import preprocessing
import Gaidel_Legacy.settings as settings

def scale_gray_levels(img):
    n, m = img.shape
    scaler = preprocessing.MinMaxScaler(feature_range=(0, settings.COLOR_MAX))
    return scaler.fit_transform(img.reshape(n * m, 1)).reshape(n, m)

def blur_image(img):
    return cv2.blur(img, settings.BLUR_SHAPE)