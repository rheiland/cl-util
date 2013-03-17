import os
import numpy as np
import pyopencl as cl
import msclib.clutil as clutil
import sys

class Colorize:
	def __init__(self, context, devices):
		filename = os.path.join(os.path.dirname(__file__), 'colorize.cl')
		program = clutil.createProgram(context, devices, [], filename)
		self.context = context
		self.kernColorizef = cl.Kernel(program, 'colorizef')
		self.kernColorizef_sat = cl.Kernel(program, 'colorizef_sat')
		self.kernColorizei = cl.Kernel(program, 'colorizei')

	def colorize(self, queue, arrIn, val=None, hue=None, typeIn=np.float32, dOut=None):
		copy_back = False

		if type(arrIn) == cl.Buffer or type(arrIn) == cl.GLBuffer:
			dIn = arrIn
			type_size = np.dtype(np.float32).itemsize
			size = arrIn.size/type_size

			if val == None:
				val = (0, 1000)

		else:
			cm = cl.mem_flags
			dIn = cl.Buffer(self.context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=arrIn)
			size = arrIn.size

			if val == None:
				val = (arrIn.min(), arrIn.max())

			typeIn = arrIn.dtype

			if dOut == None:
				copy_back = True

		if hue == None:
			hue = (0, 300)

		if dOut == None:
			dOut = dIn

		lWorksize = (256, )
		gWorksize = clutil.roundUp((size, ), lWorksize)



		if typeIn == np.float32:
			if type(hue) == int or type(hue) == float:
				self.kernColorizef_sat(queue, gWorksize, lWorksize, dIn, np.float32(val[0]), np.float32(val[1]), np.int32(size), dOut, np.float32(hue)).wait()
			if type(hue) == tuple and len(hue) == 2:
				self.kernColorizef(queue, gWorksize, lWorksize, dIn,
					np.float32(val[0]), np.float32(val[1]), np.int32(size), dOut, np.float32(hue[0]), np.float32(hue[1])).wait()
		elif typeIn == np.uint32 or typeIn == np.int32:
			if type(hue) == int or type(hue) == float:
				self.kernColorizei_sat(queue, gWorksize, lWorksize, dIn, np.int32(val[0]), np.int32(val[1]), np.int32(size), dOut, np.float32(hue)).wait()
			if type(hue) == tuple and len(hue) == 2:
				self.kernColorizei(queue, gWorksize, lWorksize, dIn,
					np.int32(val[0]), np.int32(val[1]), np.int32(size), dOut, np.float32(hue[0]), np.float32(hue[1])).wait()
		else:
			raise  TypeError("Input array can only be np.float32, np.int32 or np.uint32")

		if copy_back:
			cl.enqueue_copy(queue, arrIn, dOut)
