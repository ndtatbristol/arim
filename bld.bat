"%PYTHON%" setup.py install
if errorlevel 1 exit 1

"%PYTHON%" setup.py --version > __conda_version__.txt
if errorlevel 1 exit 1
