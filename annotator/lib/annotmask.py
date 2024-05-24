import cv2
import numpy as np

MID_POINT = (2048, 2048)

# Expected shape: (h,w,1)
def get_sqround_mask(mask):

    ### mask generation part
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    box = newbox(box)

    rdmask = np.zeros(mask.shape, np.uint8)
    rdmask = cv2.circle(rdmask, MID_POINT, 1500, (255), -1)
    cv2.drawContours(rdmask, [box], 0, (255, 255, 255), -1)

    finalmask = cv2.bitwise_and(mask, rdmask)

    return finalmask

def newbox(box):
    rate = 0.35
    keskp = (2048, 2048)
    p1 = [0, 0]
    p2 = [0, 0]
    #  find the relevant points
    if distance(box[0], box[1]) > distance(box[0], box[3]):
        p1[0] = box[0][0] + (box[1][0] - box[0][0]) / 2
        p1[1] = box[0][1] + (box[1][1] - box[0][1]) / 2
        p2[0] = box[2][0] + (box[3][0] - box[2][0]) / 2
        p2[1] = box[2][1] + (box[3][1] - box[2][1]) / 2
        if distance(keskp, p1) < distance(keskp, p2):
            i1 = 2
            i2 = 3
            # print("the points to change are 2 & 3")
        else:
            i1 = 0
            i2 = 1
            # print("the points to change are 0 & 1")
    else:
        p1[0] = box[0][0] + (box[3][0] - box[0][0]) / 2
        p1[1] = box[0][1] + (box[3][1] - box[0][1]) / 2
        p2[0] = box[2][0] + (box[1][0] - box[2][0]) / 2
        p2[1] = box[2][1] + (box[1][1] - box[2][1]) / 2
        if distance(keskp, p1) < distance(keskp, p2):
            i1 = 1
            i2 = 2
            # print("the points to change are 1 & 2")
        else:
            i1 = 3
            i2 = 0
            # print("the points to change are 0 & 3")

    # point update formulas
    i0 = basepoint(i1, i2)
    box[i1][0] = (1 - rate) * box[i0][0] + rate * box[i1][0]
    box[i1][1] = (1 - rate) * box[i0][1] + rate * box[i1][1]
    i0 = basepoint(i2, i1)
    box[i2][0] = (1 - rate) * box[i0][0] + rate * box[i2][0]
    box[i2][1] = (1 - rate) * box[i0][1] + rate * box[i2][1]

    return box


def basepoint(p1, p2):
    if p1 == 0 and p2 == 3: p0 = 1
    if p2 == 0 and p1 == 3: p0 = 2
    if p1 == 0 and p2 == 1: p0 = 3
    if p2 == 0 and p1 == 1: p0 = 2
    if p1 == 1 and p2 == 2: p0 = 0
    if p2 == 1 and p1 == 2: p0 = 3
    if p1 == 2 and p2 == 3: p0 = 1
    if p2 == 2 and p1 == 3: p0 = 0
    return p0


def distance(p1, p2):
    d = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return d
