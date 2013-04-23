import time

import pyopencl as cl
from clutil import Buffer2D
import numpy as np

platforms = cl.get_platforms()

devices = platforms[0].get_devices()
devices = [devices[1]]
context = cl.Context(devices)

cm = cl.mem_flags

dim = (800, 608)

dStrokes = Buffer2D(context, cm.READ_WRITE, dim, dtype=np.uint8)

from Brush import Brush
brush = Brush(context, devices, dStrokes)

iterations = 10

t0 = time.clock()
for i in range(iterations):
    brush.draw((0, 0), (100, 100))
dt = (time.clock() - t0) / iterations * 1000

print "%.2f milliseconds per iteration (mean)" % dt
