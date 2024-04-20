import re
import sass


def compile_scss_into_qss(scss_file_path: str, qss_file_path: str, output_style='expanded'):
    qss_str = sass.compile(filename=scss_file_path, output_style=output_style)
    with open(qss_file_path, 'w') as f:
        f.write(qss_str)


def parse_qss_by_class_name(qss_string: str, class_name: str) -> str:
    regex_str = class_name + "[^\n][^\*]*?\}"
    matches = re.findall(fr"{regex_str}", qss_string)
    qss_res_str = ""
    for m in matches:
        qss_res_str += m + "\n\n"
    return qss_res_str
