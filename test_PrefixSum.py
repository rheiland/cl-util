import os
import numpy as np
import pyopencl as cl
from PrefixSum import PrefixSum

szInt = 4

platforms = cl.get_platforms()

devices = platforms[0].get_devices()
devices = [devices[1]]
context = cl.Context(devices)
queue = cl.CommandQueue(context)

nSamples = 169740
prefixSum = PrefixSum(context, devices, nSamples)

hList = np.random.randint(0, 20, nSamples).astype(np.int32)
dList = prefixSum.factory()
cl.enqueue_copy(queue, dList, hList).wait()

hTotal = np.empty((1, ), np.int32)
dTotal = cl.Buffer(context, cl.mem_flags.READ_WRITE, 1*szInt)

prefixSum.scan(dList, dTotal, nSamples)
cl.enqueue_copy(queue, hTotal, dTotal).wait()

hTmp = np.empty((nSamples,), np.int32)
cl.enqueue_copy(queue, hTmp, dList).wait()

cl.enqueue_copy(queue, hTotal, dTotal).wait()
length = hTotal[0]

#check for correctness
assert(hList.sum() == hTotal)

print hTmp
print hList
print hTotal

#measure performance
import time
iterations = 100

cl.enqueue_copy(queue, dList, hList).wait()

t = elapsed = 0
for i in range(iterations):
    t = time.time()
    prefixSum.scan(dList, dTotal, nSamples)
    elapsed += time.time()-t

print "%.2f milliseconds per iteration (mean)" % (elapsed / iterations * 1000)

True