import os
import numpy as np
import pyopencl as cl
from IncrementalTileList import IncrementalTileList, Operator
from PrefixSum import PrefixSum
from StreamCompact import StreamCompact

szFloat =  4
szInt = 4
szChar = 1
cm = cl.mem_flags

platforms = cl.get_platforms()

devices = platforms[0].get_devices()
devices = [devices[1]]
context = cl.Context(devices)
queue = cl.CommandQueue(context)

global_dim = (4096, 4096)
global_shape = (global_dim[1], global_dim[0])

tileList = IncrementalTileList(context, devices, global_dim, (16, 16))
tiles_dim = tileList.dim
n_tiles = tiles_dim[0]*tiles_dim[1]

prefixSum = PrefixSum(context, devices, n_tiles)
streamCompact = StreamCompact(context, devices, n_tiles)

hTiles = np.random.randint(0, 20, (tiles_dim[1], tiles_dim[0])).astype(np.int32)
cl.enqueue_copy(queue, tileList.d_tiles, hTiles).wait()

tileList.build(Operator.GTE, 10)

hList = np.empty((tiles_dim[0]*tiles_dim[1],), np.int32)
cl.enqueue_copy(queue, hList, tileList.d_list).wait()

#Test correctness using tileList - prefixsum and streamcompact are then
#correct too
compact_cpu = np.where(hTiles >= 10)
compact_cpu = map(lambda x, y: y*tiles_dim[0] + x, compact_cpu[1], compact_cpu[0])
assert(np.all(compact_cpu == hList[0:tileList.length]))

import time
iterations = 100

t = elapsed = 0
for i in range(iterations):
    t = time.time()
    prefixSum.scan(dList, dTotal, n_tiles)
    elapsed += time.time()-t

print "Prefixs-sum: %.2f milliseconds per iteration (mean)" % (elapsed /
                                                           iterations
                                                        * 1000)


t = elapsed = 0
for i in xrange(iterations):
    cl.enqueue_copy(queue, dFlags, dFlags).wait()

    t = time.time()

    streamCompact.compact(dFlags, dList, dLength, n_tiles)

    elapsed += time.time()-t
print "Stream-compact: %.2f milliseconds per iteration (mean)" % (elapsed /
                                                           iterations * 1000)


t = elapsed = 0
for i in xrange(iterations):
    t = time.time()

    tileList.build(Operator.GTE, 10)

    elapsed += time.time()-t
print "Incremental tile list: %.2f milliseconds per iteration (mean)" % (
    elapsed /
                                                           iterations * 1000)

True