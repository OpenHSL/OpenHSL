# -*- coding: utf-8 -*-
import numpy
import cv2

BLUR_SIZE = 3
THRESHOLD_BLOCK_SIZE = 13


def detect_object(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, BLUR_SIZE)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img)
    val, img = cv2.threshold(img, 0, max_val, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return (0, 0), (img.shape[1], img.shape[0])
    contours = numpy.concatenate(contours)
    x, y, w, h = cv2.boundingRect(contours)
    return (x, y), (x + w, y + h)


class Splitter(object):

    def __init__(self):
        self.spectrum_rect = None

    def split(self, img):
        if self.spectrum_rect is None:
            self.detect_boxes(img)
        return img[self.spectrum_rect[0][0] : self.spectrum_rect[0][1],
                   self.spectrum_rect[1][0] : self.spectrum_rect[1][1]]

    def detect_boxes(self, img):
        self.spectrum_rect = detect_object(img)
