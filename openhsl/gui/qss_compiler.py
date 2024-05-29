import sass

qss_str = sass.compile(filename="device/Resources/Dark.scss", output_style='expanded')
with open("device/Resources/Dark.qss", 'w') as f:
    f.write(qss_str)