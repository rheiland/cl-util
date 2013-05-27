import os
import numpy as np
import pyopencl as cl
from clutil import createProgram, pow2gt, roundUp

szFloat = 4
szInt = 4
szChar = 1
cm = cl.mem_flags

LEN_WORKGROUP = 256
ELEMENTS_PER_THREAD = 2
ELEMENTS_PER_WORKGROUP = ELEMENTS_PER_THREAD*LEN_WORKGROUP

PROFILE_GPU = True

class PrefixSum:
    def __init__(self, context, devices, capacity):
        self.context = context

        if PROFILE_GPU == True:
            self.queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
        else:
            self.queue = cl.CommandQueue(context)

        filename = os.path.join(os.path.dirname(__file__), 'prefixsum.cl')
        program = createProgram(context, devices, [], filename)

        self.kernScan_pad_to_pow2 = cl.Kernel(program, 'scan_pad_to_pow2')
        self.kernScan_subarrays = cl.Kernel(program, 'scan_subarrays')
        self.kernScan_inc_subarrays = cl.Kernel(program, 'scan_inc_subarrays')

        self.lw = (LEN_WORKGROUP, )

        self.capacity = roundUp(capacity, ELEMENTS_PER_WORKGROUP)

        self.d_parts = []

        len = self.capacity/ELEMENTS_PER_WORKGROUP

        while len > 0:
            self.d_parts.append(cl.Buffer(context, cl.mem_flags.READ_WRITE, szInt*len))

            len = len/ELEMENTS_PER_WORKGROUP

        self.elapsed = 0

    def factory(self, length=None):
        if length == None:
            length = self.capacity
        elif length > self.capacity:
            raise ValueError('length > self.capacity: {0}, {1}'.format(length, self.capacity))

        length = pow2gt(length)

        return cl.Buffer(self.context, cl.mem_flags.READ_WRITE, length*szInt)

    def scan(self, dArray, dTotal, length):
        if length == None:
            length = dArray.size/szInt

        k = (length + ELEMENTS_PER_WORKGROUP - 1) / ELEMENTS_PER_WORKGROUP
        gw = (k*LEN_WORKGROUP, )

        if k == 1:
            event = self.kernScan_pad_to_pow2(self.queue, gw, self.lw,
                dArray,
                cl.LocalMemory(ELEMENTS_PER_WORKGROUP*szInt),
                np.int32(length),
                dTotal
            )
            event.wait()
            if PROFILE_GPU == True:
                self.elapsed += (event.profile.end - event.profile.start)
        else:
            if length > self.capacity:
                raise ValueError('length > self.capacity: {0}, {1}'.format(length, self.capacity))
            else:
                i = int(np.log(length)/np.log(ELEMENTS_PER_WORKGROUP))-1
                d_part = self.d_parts[i]

            event = self.kernScan_subarrays(self.queue, gw, self.lw,
                dArray,
                cl.LocalMemory(ELEMENTS_PER_WORKGROUP*szInt),
                d_part,
                np.int32(length),
            )
            event.wait()
            if PROFILE_GPU == True:
                self.elapsed += (event.profile.end - event.profile.start)

            self.scan(d_part, dTotal, k)

            event = self.kernScan_inc_subarrays(self.queue, gw, self.lw,
                dArray,
                cl.LocalMemory(ELEMENTS_PER_WORKGROUP*szInt),
                d_part,
                np.int32(length),
            )
            event.wait()
            if PROFILE_GPU == True:
                self.elapsed += (event.profile.end - event.profile.start)
