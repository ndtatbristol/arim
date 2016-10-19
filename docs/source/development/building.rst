========
Building
========

**Compiler:** The C/C++ extensions written for arim can be compiled with Visual C++ 2015. No other compiler was tested.

**Required libraries:** OpenMP 2.0 (embedded with Visual C++ 2015)

**To build arim:** in a command prompt in arim root directory::

  activate arim-dev
  python setup.py build_ext --inplace


Remark: your compiler must be in your system path. In Windows, you might want to run these commands in the Visual Studio Command Prompt.
