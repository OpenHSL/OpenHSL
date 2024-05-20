import copy

import cv2 as cv
import discorpy.post.postprocessing as post
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


def apply_barrel_distortion(image_in: np.ndarray, coeffs: np.ndarray, powers: np.ndarray, factors: np.ndarray,
                            center_xy: Optional[np.ndarray] = None, line_step: int = 40, line_val: float = 0.25
                            ) -> np.ndarray:
    image_src = image_in.astype(np.float32) / 255.0
    height, width = image_src.shape[0:2]
    xcenter = width // 2
    ycenter = height // 2

    if center_xy is not None:
        if len(center_xy) == 2:
            xcenter, ycenter = center_xy

    line_offset_x = 50
    line_offset_y = 50
    line_count_x = int(np.ceil((width - 2 * line_offset_x - width // 2 + xcenter) // line_step) // 2)
    line_count_y = int(np.ceil((height - 2 * line_offset_y - height // 2 + ycenter) // line_step) // 2)

    # Create a line-pattern image
    image_grid = np.zeros((height, width), dtype=np.float32)

    for i in range(-line_count_y, line_count_y):
        y = int(ycenter + i * line_step)
        image_grid[y - 1:y + 1] = line_val
    for i in range(-line_count_x, line_count_x):
        x = int(xcenter + i * line_step)
        image_grid[:, x - 1:x + 1] = line_val

    pad = max(width // 2, height // 2)  # Need padding as lines are shrunk after warping.
    image_grid_padded = np.pad(image_grid, pad, mode='edge')

    # Prepare powers and coeffs arrays
    j = 0
    powers_prep = []
    coeffs_prep = []
    factors_prep = np.array([10 ** f for f in factors])
    for i in range(powers[-1] + 1):
        if i in powers:
            powers_prep.append(factors_prep[j])
            coeffs_prep.append(coeffs[j])
            j += 1
        else:
            powers_prep.append(0)
            coeffs_prep.append(0)
    powers_prep = np.array(powers_prep)
    coeffs_prep = np.array(coeffs_prep)

    list_ffact = powers_prep * coeffs_prep
    image_grid_warped = post.unwarp_image_backward(image_grid_padded, xcenter + pad, ycenter + pad, list_ffact)
    image_grid_warped = image_grid_warped[pad:pad + height, pad:pad + width]
    image_in_grid_warped = copy.deepcopy(image_src)

    if len(image_src.shape) == 3:
        for i in range(3):
            image_in_grid_warped[:, :, i] = image_in_grid_warped[:, :, i] + 0.5 * image_grid_warped
        image_in_grid_warped[image_in_grid_warped > 1] = 1
    else:
        image_in_grid_warped = image_src + 0.5 * image_grid_warped

    image_in_grid_warped = (image_in_grid_warped * 255).astype(np.uint8)

    return image_in_grid_warped


def compile_scss_into_qss(scss_file_path: str, qss_file_path: str, output_style='expanded'):
    qss_str = sass.compile(filename=scss_file_path, output_style=output_style)
    with open(qss_file_path, 'w') as f:
        f.write(qss_str)


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
