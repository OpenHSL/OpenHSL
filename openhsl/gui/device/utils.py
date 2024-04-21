import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PyQt6.QtGui import QImage, QPixmap
import re
import sass


def compile_scss_into_qss(scss_file_path: str, qss_file_path: str, output_style='expanded'):
    qss_str = sass.compile(filename=scss_file_path, output_style=output_style)
    with open(qss_file_path, 'w') as f:
        f.write(qss_str)


# https://stackoverflow.com/a/32085761
def latex_to_pixmap(latex_expression: str, font_size: int, color: str) -> QPixmap:
    fig = mpl.figure.Figure()
    fig.patch.set_facecolor('none')
    fig.set_canvas(FigureCanvasAgg(fig))
    renderer = fig.canvas.get_renderer()

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
    pixmap = QPixmap(image)

    return pixmap


def parse_qss_by_class_name(qss_string: str, class_name: str) -> str:
    regex_str = class_name + "[^\n][^\*]*?\}"
    matches = re.findall(fr"{regex_str}", qss_string)
    qss_res_str = ""
    for m in matches:
        qss_res_str += m + "\n\n"
    return qss_res_str
