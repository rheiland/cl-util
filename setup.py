__author__ = 'marcdeklerk'

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [
    #    Extension("Tool", ["Tool.pyx"], include_dirs=[numpy.get_include()]),
    Extension("Brush", ["Brush.pyx"], include_dirs=[numpy.get_include()]),
    Extension("PrefixSum", ["PrefixSum.pyx"],
        include_dirs=[numpy.get_include()]),
    Extension("StreamCompact", ["StreamCompact.pyx"],
        include_dirs=[numpy.get_include()]),
    Extension("IncrementalTileList", ["IncrementalTileList.pyx"],
        include_dirs=[numpy.get_include()]),
    Extension("GraphCut", ["GraphCut.pyx"], include_dirs=[numpy.get_include(
    )]),
]

setup(
    name='Brush',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
