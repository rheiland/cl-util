# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False

import os
import numpy as np
import pyopencl as cl
from clutil import createProgram, roundUp, pow2gt

from libc.math cimport log

DEF szFloat = 4
DEF szInt = 4
DEF szChar = 1
cm = cl.mem_flags

DEF LEN_WORKGROUP = 256
DEF ELEMENTS_PER_THREAD = 2
DEF ELEMENTS_PER_WORKGROUP = ELEMENTS_PER_THREAD*LEN_WORKGROUP

cdef class PrefixSum:
    cdef:
        int capacity
        object context
        object queue
        list d_parts
        tuple lw

        object kernScan_pad_to_pow2
        object kernScan_subarrays
        object kernScan_inc_subarrays

    def __cinit__(self, context, devices, int capacity):
        self.capacity = capacity

    def __init__(self, context, devices, int capacity):
        self.context = context
        self.queue = cl.CommandQueue(context)

        filename = os.path.join(os.path.dirname(__file__), 'scan/harris/scan.cl')
        program = createProgram(context, devices, [], filename)

        self.kernScan_pad_to_pow2 = cl.Kernel(program, 'scan_pad_to_pow2')
        self.kernScan_subarrays = cl.Kernel(program, 'scan_subarrays')
        self.kernScan_inc_subarrays = cl.Kernel(program, 'scan_inc_subarrays')

        self.lw = (LEN_WORKGROUP, )

        self.capacity = roundUp(capacity, ELEMENTS_PER_WORKGROUP)

        self.d_parts = []

        len = self.capacity/ELEMENTS_PER_WORKGROUP

        while len > 1:
            self.d_parts.append(cl.Buffer(context, cl.mem_flags.READ_WRITE, szInt*len))

            len = len/ELEMENTS_PER_WORKGROUP

    cpdef factory(self, length=None):
        if length == None:
            length = self.capacity
        elif length > self.capacity:
            raise ValueError('length > self.capacity: {0}, {1}'.format(length, self.capacity))

        length = pow2gt(length)

        return cl.Buffer(self.context, cl.mem_flags.READ_WRITE, length*szInt)

    #array size must be <= workgroupsize*2
    cpdef scan(self, dArray, dTotal, int length):
        if length == None:
            length = dArray.size/szInt

        cdef int i
        cdef int k = (length + ELEMENTS_PER_WORKGROUP - 1) / ELEMENTS_PER_WORKGROUP
        cdef tuple gw = (k*LEN_WORKGROUP, )
        cdef object d_part

        if k == 1:
            self.kernScan_pad_to_pow2(self.queue, gw, self.lw,
                dArray,
                cl.LocalMemory(ELEMENTS_PER_WORKGROUP*szInt),
                np.int32(length),
                dTotal
            ).wait()
        else:
            if length > self.capacity:
                raise ValueError('length > self.capacity: {0}, {1}'.format(length, self.capacity))
            else:
                i = <int> (log(length)/log(ELEMENTS_PER_WORKGROUP))-1
                d_part = self.d_parts[i]

            self.kernScan_subarrays(self.queue, gw, self.lw,
                dArray,
                cl.LocalMemory(ELEMENTS_PER_WORKGROUP*szInt),
                d_part,
                np.int32(length),
            ).wait()

            self.scan(d_part, dTotal, k)

            self.kernScan_inc_subarrays(self.queue, gw, self.lw,
                dArray,
                cl.LocalMemory(ELEMENTS_PER_WORKGROUP*szInt),
                d_part,
                np.int32(length),
            ).wait()
#
