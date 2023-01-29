import cv2
import Gaidel_Legacy.settings as settings

def blur_image(img):
    return cv2.blur(img, settings.BLUR_SHAPE)