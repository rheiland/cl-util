from CLCanvas import Filter as CLFilter

import os, sys
import numpy as np
import pyopencl as cl
from clutil import roundUp, createProgram, compareFormat, isFormat, Buffer2D

LWORKGROUP = (16, 16)

class Colorize():
	class HUES():
		STANDARD = (0, 240)
		REVERSED = (240, 0)

	class SATURATION():
		STANDARD = 0

	class Filter(CLFilter):
		def __init__(self, kern, format, range, hues, vals, sats):
			self.kern = kern
			self.format = format
			self.range = range
			self.hues = hues
			self.vals = vals
			self.sats = sats

		def execute(self, queue, args):
			if self.format[0] == Buffer2D:
				buf = args[-1]
				args.append(np.array(buf.dim, np.int32))

			args += [
				np.array(self.range, np.float32),
				np.array(self.hues, np.float32),
				np.array(self.vals, np.float32),
				np.array(self.sats, np.float32)
				]

			gw = buf.dim

			self.kern(queue, gw, LWORKGROUP, *args)

	def __init__(self, canvas):
		filename = os.path.join(os.path.dirname(__file__), 'colorize.cl')
		program = createProgram(canvas.context, canvas.devices, [], filename)

		self.kernels_i32 = cl.Kernel(program, 'colorize_i32')
#		self.kernels_f32 = cl.Kernel(program, 'colorize_f32')
		self.kernels_f32 = cl.Kernel(program, 'val_f32')

	def factory(self, format, range, hues=None, sats=None, vals=None):
		if compareFormat(format, (Buffer2D, np.int32)):
			kern = self.kernels_i32;
		elif compareFormat(format, (Buffer2D, np.float32)):
			if hues == None:
				hues = Colorize.HUES.STANDARD
			if sats == None:
				sats = [1, 1]
			if vals == None:
				vals = [1, 1]

			kern = self.kernels_f32;

		return Colorize.Filter(kern, format, range, hues, sats, vals)