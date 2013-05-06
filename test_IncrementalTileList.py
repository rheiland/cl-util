import os
import numpy as np
import pyopencl as cl
from clutil import Buffer2D, roundUp, createProgram
from IncrementalTileList import IncrementalTileList, Operator, Logical

szFloat =  4
szInt = 4
szChar = 1
cm = cl.mem_flags

platforms = cl.get_platforms()

devices = platforms[0].get_devices()
devices = [devices[1]]
context = cl.Context(devices)
queue = cl.CommandQueue(context)

dim = (800, 608)
shape = (dim[1], dim[0])
nSamples = dim[0]*dim[1]

tileList = IncrementalTileList(context, devices, dim, (16, 16))

hTiles = np.random.randint(0, 20, shape).astype(np.int32)
cl.enqueue_copy(queue, tileList.d_tiles, hTiles).wait()

tileList.build(Operator.GTE, 10)

hList = np.empty((nSamples,), np.int32)
cl.enqueue_copy(queue, hList, tileList.d_list).wait()

print hTiles
print 'dim: {0}, num elements: {1}'.format(dim, dim[0]*dim[1])

print hList

#test correctness
compact_cpu = np.where(hTiles >= 10)
compact_cpu = map(lambda x, y: y*dim[0] + x, compact_cpu[1], compact_cpu[0])
assert(np.all(compact_cpu == hList[0:tileList.length]))

import time
iterations = 100

t = elapsed = 0
for i in xrange(iterations):
    t = time.time()

    tileList.build(Operator.GTE, 10)

    elapsed += time.time()-t
print "%.2f milliseconds per iteration (mean)" % (elapsed / iterations * 1000)

True