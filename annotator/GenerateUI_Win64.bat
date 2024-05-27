@echo off

REM Run me in Anaconda Prompt with the corresponding environment enabled!

echo Running pyuic5...
call pyuic5 ui\datmant.ui -o ui\datmant_ui.py
call pyuic5 ui\color_specs.ui -o ui\color_specs_ui.py
echo Done.