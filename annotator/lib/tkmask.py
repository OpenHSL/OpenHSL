import cv2
import numpy as np
import shapefile
import os
from PyQt5.QtGui import QColor

SHAPETYPES = ['KPIKIPR', 'KVUUK', 'PAIK_J', 'POIKPR', 'SERV', 'VORK', 'PAIK', 'MUREN', 'AUK']


# Produces tehnokeskuse defect mask as the helper layer
def generate_tk_defects_layer(path, shpath, fname, colordefs, warning, log=print):

    # The shape file is assumed to be one directory up than the orthophotos
    path = path.strip("\\")  # Remove trailing slash
    path += os.path.sep  # Reintroduce trailing slash

    try:
        mask = cv2.imread(path + fname + '.mask.png', 0)
    except Exception as e:
        raise OSError("Cannot read the orthoframe mask file. " + \
                      "Make sure it exists in the image directory and has the extension .mask.png")

    # Need to generate an empty transparent image
    h, w = mask.shape[:2]
    img = np.zeros((h,w,4), 'uint8')

    # read the vrt parameters
    koord = runvrt(path + fname + '.vrt')

    xmin = koord[0]
    xmax = koord[0] + koord[1] * (w - 1)
    ymin = koord[3] + koord[5] * (h - 1)
    ymax = koord[3]

    pnts, tyyp = getdefects(shpath, xmin, xmax, ymin, ymax, koord, warning, log=log)

    has_warning = False

    for i in range(0, len(tyyp)):

        pp = np.asarray(pnts[i], dtype=np.int32)

        if SHAPETYPES[tyyp[i]] in colordefs:
            col = colordefs[SHAPETYPES[tyyp[i]]]
            rgb = list(QColor(col).getRgb())[:-1]
            rgb.append(99)

            if tyyp[i] < 5:  # joondefektid
                cv2.polylines(img, [pp], False, rgb, 40)
            if 4 < tyyp[i] < 8:  # pinddefektid
                cv2.fillPoly(img, [pp], rgb)
            if tyyp[i] == 8:
                cv2.circle(img, pnts[i][0], 50, rgb, 25)
        else:
            log("Cannot locate " + SHAPETYPES[tyyp[i]] + " in color definitions, skipping!")
            has_warning = True

    if has_warning:
        warning.append("problem with color definitions")

    # Mask away pixels
    img[mask==0] = (0, 0, 0, 0)

    return img


def getdefects(path, xmin, xmax, ymin, ymax, koord, warning, log=print):
    deflist = ['defects_polygon', 'defects_line', 'defects_point']

    cnt = np.zeros((9,), dtype=int)
    points = []
    rike = []
    k = 0

    # has_warning = False

    for j in range(3):

        try:
            kuju = shapefile.Reader(path + deflist[j])  # three separate defect files

            for i in range(len(kuju.shapes())):
                shape_ex = kuju.shape(i)
                if j == 2:  # point
                    x1 = x2 = shape_ex.points[0][0]
                    y1 = y2 = shape_ex.points[0][1]
                else:  # not point type defect
                    x1 = shape_ex.bbox[0]
                    x2 = shape_ex.bbox[2]
                    y1 = shape_ex.bbox[1]
                    y2 = shape_ex.bbox[3]

                    # combinations, I guess

                # if x1>xmin and x2<xmax and y1>ymin and y2<ymax:
                # all defect points are within the image
                # any defect point is within the image
                if xmin < x1 < xmax and ymin < y1 < ymax or xmin < x1 < xmax and ymin < y2 < ymax or xmin < x2 < xmax and ymin < y1 < ymax or xmin < x2 < xmax and ymin < y2 < ymax:

                    points.append([])  # = np.zeros((len(shape_ex.points),1))  # a vector of zeroes
                    rike.append([])

                    for ip in range(len(shape_ex.points)):
                        x = int(round((shape_ex.points[ip][0] - koord[0]) / koord[1]))
                        y = int(round((shape_ex.points[ip][1] - koord[3]) / koord[5]))

                        points[k].append((x, y))
                    # now the defect points are in points

                    rec = kuju.shapeRecord(i)
                    indices = [l for l, s in enumerate(SHAPETYPES) if
                               rec.record[2] == s]  # the index of the defect type, isnt indices a scalar?
                    cnt[indices[0]] += 1  # cnt is a summary over the image
                    rike[k] = indices[0]
                    k += 1

        except Exception as e:
            log("The defect list " + deflist[j] + " could not be found and will be skipped")
            # has_warning = True

    # Issue warning about missing definition lists # TODO: Not sure how important this is, will disable by default
    # if has_warning:
        #warning.append("some defect lists were missing from the database")

    return points, rike


def runvrt(fname):
    try:
        vrtfile = open(fname, 'r')
        for line in vrtfile:
            if line.find('<GeoTransform>') > -1:
                koord = line[line.find('<GeoTransform>') + 16:line.find('</GeoTransform>')]
                break
        vrtfile.close()
        koord = ''.join(koord)
        koord = np.fromstring(koord, dtype=np.float, sep=',')
        return koord
    except Exception as e:
        raise OSError("Could not open the .vrt file. Make sure it exists in the image directory.")
