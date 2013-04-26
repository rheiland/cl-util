import os
import numpy as np
import pyopencl as cl
from clutil import Buffer2D, roundUp, createProgram
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

nSamples = 169740
capcity = nSamples

streamCompact = StreamCompact(context, devices, capcity)

hList = np.empty((nSamples,), np.int32)
dList = streamCompact.listFactory(nSamples)

hFlags = np.random.randint(0, 2, nSamples).astype(np.int32)
dFlags = streamCompact.flagFactory(nSamples)
cl.enqueue_copy(queue, dFlags, hFlags).wait()

hLength = np.empty((1, ), np.int32)
dLength = cl.Buffer(context, cl.mem_flags.READ_WRITE, 1*szInt)

streamCompact.compact(dFlags, dList, dLength, nSamples)
cl.enqueue_copy(queue, hList, dList).wait()
cl.enqueue_copy(queue, hLength, dLength).wait()

print 'flags', hFlags

#test correctness
compact_cpu = np.where(hFlags == 1)[0]
assert(np.all(compact_cpu == hList[0:hLength]))

import time
iterations = 100

t = elapsed = 0
for i in xrange(iterations):
    cl.enqueue_copy(queue, dFlags, dFlags).wait()

    t = time.time()

    streamCompact.compact(dFlags, dList, dLength, nSamples)

    elapsed += time.time()-t
print "%.2f milliseconds per iteration (mean)" % (elapsed / iterations * 1000)

True