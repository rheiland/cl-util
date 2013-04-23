import os
import numpy as np
import pyopencl as cl
from clutil import createProgram, roundUp, ceil_divi, pow2gt

szFloat = np.dtype(np.float32).itemsize
szInt = np.dtype(np.int32).itemsize
szChar = np.dtype(np.uint8).itemsize
cm = cl.mem_flags

LEN_WORKGROUP = 256
ELEMENTS_PER_THREAD = 2

ELEMENTS_PER_WORKGROUP = ELEMENTS_PER_THREAD*LEN_WORKGROUP

DEBUG = False

class PrefixSum:
	def __init__(self, context, devices, capacity):
		filename = os.path.join(os.path.dirname(__file__), 'scan/harris/scan.cl')
		program = createProgram(context, devices, [], filename)
		self.context = context
		self.kernScan_pad_to_pow2 = cl.Kernel(program, 'scan_pad_to_pow2')
		self.kernScan_subarrays = cl.Kernel(program, 'scan_subarrays')
		self.kernScan_inc_subarrays = cl.Kernel(program, 'scan_inc_subarrays')

		self.lw = (LEN_WORKGROUP, )

		self.capacity = roundUp(capacity, ELEMENTS_PER_WORKGROUP)

		nBytes = szInt*capacity

		self.dParts = []
		len = self.capacity/ELEMENTS_PER_WORKGROUP
		while len > 1:
			self.dParts += [cl.Buffer(context, cl.mem_flags.READ_WRITE, szInt*len)]

			len = len/ELEMENTS_PER_WORKGROUP

	def factory(self, length=None):
		if length == None:
			length = self.capacity
		elif length > self.capacity:
			raise ValueError('length > self.capacity: {0}, {1}'.format(length, self.capacity))

		length = pow2gt(length)

		return cl.Buffer(self.context, cl.mem_flags.READ_WRITE, length*szInt)

	#array size must be <= workgroupsize*2
	def scan(self, queue, dArray, dTotal, length=None,):
		if length == None:
			length = dArray.size/szInt

		k = ceil_divi(length, ELEMENTS_PER_WORKGROUP)
		gw = (k*LEN_WORKGROUP, )

		if DEBUG:
			print 'scan length {0}, {1} parts'.format(length, k)
			hPart = np.empty((k, ), np.int32)

			hArray = np.empty((length), np.int32)
			cl.enqueue_copy(queue, hArray, dArray)

			print 'array:'
			print hArray

		if k == 1:
			args = [
				dArray,
				cl.LocalMemory(ELEMENTS_PER_WORKGROUP*szInt),
				np.int32(length),
				dTotal
			]
			self.kernScan_pad_to_pow2(queue, gw, self.lw, *args).wait()

			if DEBUG:
				hTotal = np.empty((1,), np.int32)
				cl.enqueue_copy(queue, hTotal, dTotal)

				print 'total: {0}'.format(hTotal[0])
		else:
			if length > self.capacity:
				raise ValueError('length > self.capacity: {0}, {1}'.format(length, self.capacity))
			else:
				i = int(np.log(length)/np.log(ELEMENTS_PER_WORKGROUP))-1
				dPart = self.dParts[i]

			args = [
				dArray,
				cl.LocalMemory(ELEMENTS_PER_WORKGROUP*szInt),
				dPart,
				np.int32(length),
			]

			self.kernScan_subarrays(queue, gw, self.lw, *args).wait()

			if DEBUG:
				cl.enqueue_copy(queue, hPart, dPart)
				print 'parts:'
				print hPart

			self.scan(queue, dPart, dTotal, k)

			self.kernScan_inc_subarrays(queue, gw, self.lw, *args).wait()

if __name__ == "__main__":
	DEBUG = True

	platforms = cl.get_platforms()

	devices = platforms[0].get_devices()
	device = devices[1]
	context = cl.Context([device])
	queue = cl.CommandQueue(context,properties=cl.command_queue_properties.PROFILING_ENABLE)

	prefixSum = PrefixSum(context, [device], 16974)
	dList = prefixSum.factory()

	hTotal = np.empty((1, ), np.int32)
	dTotal = cl.Buffer(context, cl.mem_flags.READ_WRITE, 1*szInt)

	nSamples = 16974
	hList = np.random.randint(0, 20, nSamples).astype(np.int32)
	cl.enqueue_copy(queue, dList, hList)

	prefixSum.scan(queue, dList, dTotal, nSamples)
	cl.enqueue_copy(queue, hTotal, dTotal)

	hTmp = np.empty((nSamples,), np.int32)
	cl.enqueue_copy(queue, hTmp, dList)

	cl.enqueue_copy(queue, hTotal, dTotal)
	length = hTotal[0]

	assert(hList.sum() == hTotal)

	True