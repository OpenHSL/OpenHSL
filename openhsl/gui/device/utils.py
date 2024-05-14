import cv2 as cv
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from PyQt6.QtCore import QRect
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication
import re
import sass
from typing import Dict, List, Optional, Tuple, Union


def compile_scss_into_qss(scss_file_path: str, qss_file_path: str, output_style='expanded'):
    qss_str = sass.compile(filename=scss_file_path, output_style=output_style)
    with open(qss_file_path, 'w') as f:
        f.write(qss_str)


def detect_corners(image: np.ndarray, roi: np.ndarray, sobel_kernel_size: int):
    image_t_gray = image
    ry, rx, rh, rw = roi

    if len(image.shape) == 3:
        image_t_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    image_sobel_x = np.abs(cv.Sobel(image_t_gray, cv.CV_64F, 1, 0, ksize=sobel_kernel_size))
    image_sobel_y = np.abs(cv.Sobel(image_t_gray, cv.CV_64F, 0, 1, ksize=sobel_kernel_size))
    image_sobel_xy = image_sobel_x + image_sobel_y

    image_edged = image_sobel_xy
    tl, tr, br, bl = find_corners(image_edged[ry:ry + rh, rx:rx + rw])

    for p in [tl, tr, br, bl]:
        p += [ry, rx]

    return image_edged, [tl, tr, br, bl]


def estimate_barrel_distortion_equation(image: np.ndarray, xy_center: Tuple[int, int], roi_rect: QRect,
                                        equation_dict: Dict[str, Dict[str, float]]):
    list_power = np.asarray([10 ** equation_dict['power'][f'{p}'] for p in range(len(equation_dict['power']))])
    list_coef = equation_dict['coef']
    h, w = image.shape[0:2]
    x_center, y_center = xy_center


def find_corners(image: np.ndarray):
    h, w = image.shape[0:2]
    top_left = [0, 0]
    top_right = [0, w]
    bottom_right = [h, w]
    bottom_left = [h, 0]
    coords = np.argwhere(image)

    idx = np.argmin(np.linalg.norm(coords - top_left, axis=1))
    top_left = coords[idx]

    idx = np.argmin(np.linalg.norm(coords - top_right, axis=1))
    top_right = coords[idx]

    idx = np.argmin(np.linalg.norm(coords - bottom_right, axis=1))
    bottom_right = coords[idx]

    idx = np.argmin(np.linalg.norm(coords - bottom_left, axis=1))
    bottom_left = coords[idx]

    return top_left, top_right, bottom_right, bottom_left


def latex_to_file(path: str, latex_expression: str, color: str, font_size: int = None) -> None:
    image = latex_to_image(latex_expression, color, font_size)
    image.save(path, 'png', 100)


# https://stackoverflow.com/a/32085761
def latex_to_image(latex_expression: str, color: str, font_size: int = None) -> QImage:
    font = QApplication.font()
    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = font.family()
    mpl.rcParams['mathtext.it'] = f'{font.family()}:italic'
    mpl.rcParams['mathtext.bf'] = f'{font.family()}:bold'
    # matplotlib.rcParams['font.family'] = 'Segoe UI'
    fig = Figure()
    fig.patch.set_facecolor('none')
    fig.set_canvas(FigureCanvasAgg(fig))
    renderer = fig.canvas.get_renderer()

    if font_size is None:
        font_size = font.pointSize()

    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.patch.set_facecolor('none')
    t = ax.text(0, 0, latex_expression, ha='left', va='bottom', fontsize=font_size, color=color)

    fwidth, fheight = fig.get_size_inches()
    fig_bbox = fig.get_window_extent(renderer)

    text_bbox = t.get_window_extent(renderer)

    tight_fwidth = text_bbox.width * fwidth / fig_bbox.width
    tight_fheight = text_bbox.height * fheight / fig_bbox.height

    fig.set_size_inches(tight_fwidth, tight_fheight)

    buf, size = fig.canvas.print_to_buffer()
    image = QImage.rgbSwapped(QImage(buf, size[0], size[1], QImage.Format.Format_ARGB32))

    return image


def parse_qss_by_class_name(qss_string: str, class_name: str) -> str:
    regex_str = class_name + "[^\n][^\*]*?\}"
    matches = re.findall(fr"{regex_str}", qss_string)
    qss_res_str = ""
    for m in matches:
        qss_res_str += m + "\n\n"
    return qss_res_str
