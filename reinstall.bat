python setup.py bdist_wheel
rmdir /s /q build
pip install -U --force-reinstall --no-deps -f ./dist openhsl
pause