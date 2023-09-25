from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension("hall1997_cython",
              sources=["hall1997_cython.pyx"],
              libraries=["m"]  # Unix-like specific
              )
]

setup(name="hall1997_cython",
      ext_modules=cythonize(ext_modules))