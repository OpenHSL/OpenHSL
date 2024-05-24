@echo off

pyinstaller --add-data "res/A.ico;res/" datmant.py

echo Done.