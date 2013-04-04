from CLCanvas import Filter as CLFilter

import os, sys
import numpy as np
import pyopencl as cl
from clutil import roundUp, createProgram, compareFormat, isFormat

LWORKGROUP = (16, 16)

class Colorize():
	class HUES():
		STANDARD = (0, 240)
		REVERSED = (240, 0)

	class Filter(CLFilter):
		def __init__(self, kern, format, range, hues):
			self.kern = kern
			self.format = format
			self.setRange(range)
			self.setHues(hues)

		def setRange(self, range):
			if compareFormat(self.format, (cl.Buffer, np.int32)):
				self.range = np.array(range, np.int32)
			elif compareFormat(self.format, (cl.Buffer, np.float32)):
				self.range = np.array(range, np.float32)

		def setHues(self, hues):
			self.hues = np.array(hues, np.int32)

		def execute(self, queue, args):
			if self.format[0] == cl.Buffer:
				shape = args[-1].shape
				args += [np.array(shape, np.int32)]

			gw = roundUp(shape, LWORKGROUP)

			self.kern(queue, gw, LWORKGROUP, self.range, self.hues, *args)

	def __init__(self, canvas):
		filename = os.path.join(os.path.dirname(__file__), 'colorize.cl')
		program = createProgram(canvas.context, canvas.devices, [], filename)

		self.kernels_i32 = cl.Kernel(program, 'colorize_i32')
		self.kernels_f32 = cl.Kernel(program, 'colorize_f32')

	def factory(self, format, range, hues=None):
		if compareFormat(format, (cl.Buffer, np.int32)):
			kern = self.kernels_i32;
		elif compareFormat(format, (cl.Buffer, np.float32)):
			kern = self.kernels_f32;

		if hues == None:
			hues = Colorize.HUES.STANDARD

		return Colorize.Filter(kern, format, range, hues)