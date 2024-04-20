import sass


def compile_scss_into_qss(scss_file_path: str, qss_file_path: str, output_style='expanded'):
    qss_str = sass.compile(filename=scss_file_path, output_style=output_style)
    with open(qss_file_path, 'w') as f:
        f.write(qss_str)
