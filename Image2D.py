import pyopencl as cl
from clutil import createProgram, roundUp, compareFormat
import numpy as np
import os

LWORKGROUP = (16, 16)

class Image2D(cl.Image):
    def __init__(self, context, flags, format, dim):
        cl.Image.__init__(self, context, flags, format, dim)

        self.dim = dim
