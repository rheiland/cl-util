import os
import numpy as np
import pyopencl as cl
import msclib.clutil as clutil
from clutil import Buffer2D, alignedDim
import sys
import PrefixSum

szFloat = np.dtype(np.float32).itemsize
szInt = np.dtype(np.int32).itemsize
szChar = np.dtype(np.uint8).itemsize
cm = cl.mem_flags

LEN_WORKGROUP = 256
LWORKGROUP_2D = (16, 16)

class StreamCompact():
	OPERATOR_EQUAL = 0
	OPERATOR_GT = 1
	OPERATOR_LT = 2
	OPERATOR_GTE = 3
	OPERATOR_LTE = 4

	LOGICAL_AND = 0
	LOGICAL_OR = 1

	def __init__(self, context, devices, capacity):
		self.context = context
		self.capacity = capacity

		filename = os.path.join(os.path.dirname(__file__), 'streamcompact.cl')
		program = clutil.createProgram(context, devices, [], filename)

		self.kernFlag = cl.Kernel(program, 'flag')
		self.kernFlagLogcal = cl.Kernel(program, 'flagLogical')
		self.kernCompact = cl.Kernel(program, 'compact')

		self.lw = (LEN_WORKGROUP, )

		self.prefixSum = PrefixSum.PrefixSum(context, devices, capacity)

	def flagFactory(self, length=None):
		if length == None:
			length = self.capacity
		elif length > self.capacity:
			raise ValueError('length > self.capacity: {0}, {1}'.format(length, self.capacity))

		return cl.Buffer(self.context, cl.mem_flags.READ_WRITE, length*szInt)

	def listFactory(self, length=None):
		if length == None:
			length = self.capacity
		elif length > self.capacity:
			raise ValueError('length > self.capacity: {0}, {1}'.format(length, self.capacity))

		return self.prefixSum.factory(length)

	def flag(self, queue, dIn, dOut, operator, operand, length=None):
		if length == None:
			length = dIn.size/szInt
		elif length > self.capacity:
			raise ValueError('length > self.capacity: {0}, {1}'.format(length, self.capacity))

		args = [
			dIn,
			dOut,
			np.int32(length),
			np.int32(operator),
			np.int32(operand)
		]

		gw = clutil.roundUp((length, ), self.lw)

		self.kernFlag(queue, gw, self.lw, *args).wait()

	def flagLogical(self, queue, dIn1, dIn2, dOut, operator1, operand1, operator2, operand2, logical, length=None):
		if length == None:
			length = dIn1.size/szInt
		elif length > self.capacity:
			raise ValueError('length > self.capacity: {0}, {1}'.format(length, self.capacity))

		args = [
			dIn1,
			dIn2,
			dOut,
			np.int32(length),
			np.int32(operator1),
			np.int32(operator2),
			np.int32(operand1),
			np.int32(operand2),
			np.int32(logical)
		]

		gw = clutil.roundUp((length, ), self.lw)

		self.kernFlagLogcal(queue, gw, self.lw, *args).wait()

	def compact(self, queue, dFlags, dList, dLength, length=None):
		if length == None:
			length = dFlags.size/szInt
		elif length > self.capacity:
			raise ValueError('length > self.capacity: {0}, {1}'.format(length, self.capacity))

		cl.enqueue_copy(queue, dList, dFlags)
		self.prefixSum.scan(queue, dList, dLength, length)

		args = [
			dFlags,
			dList,
			dList,
			np.int32(length)
		]

		gw = clutil.roundUp((length, ), self.lw)

		self.kernCompact(queue, gw, self.lw, *args).wait()

class TileList():
	def __init__(self, context, devices, shape):
		self.hLength = np.empty((1,), np.int32)
		self.shape = shape

		self.streamCompact = StreamCompact(context, devices, shape[0]*shape[1])

		self.dLength = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=self.hLength)
		self.dList = self.streamCompact.listFactory()

		self.queue = cl.CommandQueue(context)

		self.isDirty = True

		self.dTiles = Buffer2D.fromBuffer(self.streamCompact.flagFactory(), shape, shape)
		self.dFlags = Buffer2D.fromBuffer(self.streamCompact.flagFactory(), shape, shape)

		filename = os.path.join(os.path.dirname(__file__), 'streamcompact.cl')
		program = clutil.createProgram(context, devices, [], filename)

		self.kernInit = cl.Kernel(program, 'init')

		self.initIteration = -1

		self.reset()

	@property
	def length(self):
		if self.isDirty:
			cl.enqueue_copy(self.queue, self.hLength, self.dLength).wait()
			self.isDirty = False

		return int(self.hLength[0])

	def increment(self):
		self.iteration += 1

	def flag(self, operator=None, operand=None):
		if operator == None: operator = StreamCompact.OPERATOR_EQUAL
		if operand == None:  operand = self.iteration

		self.streamCompact.flag(self.queue, self.dTiles, self.dFlags, operator, operand)
		self.streamCompact.compact(self.queue, self.dFlags, self.dList, self.dLength)
		self.isDirty = True

	def flagLogical(self, dTiles2, operator1, operand1, operator2, operand2, logical):
		self.streamCompact.flagLogical(self.queue, self.dTiles, dTiles2, self.dFlags, operator1, operand1, operator2, operand2, logical)
		self.streamCompact.compact(self.queue, self.dFlags, self.dList, self.dLength)
		self.isDirty = True

	def reset(self):
		self.iteration = self.initIteration + 1

		args = [
			self.dTiles,
			self.dTiles.dim,
			np.int32(self.initIteration)
		]

		gw = clutil.roundUp(self.shape, LWORKGROUP_2D)

		self.kernInit(self.queue, gw, LWORKGROUP_2D, *args).wait()


if __name__ == "__main__":
	import time
	platforms = cl.get_platforms()

	devices = platforms[0].get_devices()
	device = devices[1]
	clContext = cl.Context([device])
	clQueue = cl.CommandQueue(clContext,properties=cl.command_queue_properties.PROFILING_ENABLE)

	capcity = 2**19
	streamCompact = StreamCompact(clContext, [device], capcity)

	dList = streamCompact.listFactory()
	dTiles = streamCompact.flagFactory()
	dFlags = streamCompact.flagFactory()

	hLength = np.empty((1, ), np.int32)
	dLength = cl.Buffer(clContext, cl.mem_flags.READ_WRITE, 1*szInt)

	nSamples = 16974
	hTiles = np.random.randint(0, 20, nSamples).astype(np.int32)
	cl.enqueue_copy(clQueue, dTiles, hTiles)

	iter = 1

	elapsed = 0
	t1 = t2 = 0
	for i in xrange(iter):
		cl.enqueue_copy(clQueue, dTiles, hTiles)

		t1 = time.time()

		streamCompact.flag(clQueue, dTiles, dFlags, StreamCompact.OPERATOR_GTE, 15, nSamples)
		streamCompact.compact(clQueue, dFlags, dList, dLength, nSamples)

		elapsed += time.time()-t1
	print 'compact: ', elapsed/iter

	hTmp = np.empty((nSamples,), np.int32)
	cl.enqueue_copy(clQueue, hTmp, dList)

	cl.enqueue_copy(clQueue, hLength, dLength)
	length = hLength[0]

	res = np.where(hTiles >= 15)[0] == hTmp[0:length]
	assert len(np.where(res == False)[0]) == 0

	shape = (1000, 10)
	hTiles = np.random.randint(0, 20, shape[0]*shape[1]).astype(np.int32)
	hTiles = hTiles.reshape(shape)

	tileList = TileList(clContext, [device], shape)

	cl.enqueue_copy(clQueue, tileList.dTiles, hTiles)

	tileList.flag(StreamCompact.OPERATOR_GT, 15)

	hList = np.empty((shape[0]*shape[1],), np.int32)
	cl.enqueue_copy(clQueue, hList, tileList.dList)

	res = hList.ravel()[0:tileList.length] == np.where(hTiles.ravel() > 15)[0]
	assert(res.all() == True)

	True