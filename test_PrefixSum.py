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

nSamples = 65536
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

#evaluate performance
import time
import csv
from evaluate import global_dims, iterations, tile_dim, columns

res_file = open('results/prefixsum.csv', 'w')
resWriter = csv.writer(res_file)

resWriter.writerow(columns)

for global_dim in global_dims:
    n_tiles = (global_dim[0]/tile_dim[0])*(global_dim[1]/tile_dim[1])
    prefixSum = PrefixSum(context, devices, n_tiles)

    dList = prefixSum.factory()
    dTotal = cl.Buffer(context, cl.mem_flags.READ_WRITE, 1*szInt)

    mp = float(global_dim[0]*global_dim[1])/(1024*1024)

    t = elapsed = 0
    for i in range(iterations):
        t = time.time()

        prefixSum.scan(dList, dTotal, n_tiles)
        elapsed += time.time()-t

    print "{0:.2f}mp: {1:.2f}ms per iteration (mean)".format(mp, (elapsed/iterations * 1000))
    print "{0:.2f}ms kernel time".format(1e-9 * prefixSum.elapsed)

    row = [
        "({0} {1})".format(global_dim[0], global_dim[1]),
        mp,
        "({0} {1})".format(tile_dim[0], tile_dim[1]),
        n_tiles,
        (elapsed/iterations * 1000)
    ]


    resWriter.writerow(row)

True