from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext

ext_modules = [
    Extension("ViolaJones",  ["ViolaJones.py"])
    #   ... all your modules that need be compiled ...
]
setup(
    name='Viola_Jones_Eye_Detection',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
