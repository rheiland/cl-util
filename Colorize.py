from CLCanvas import ChainedFilter

import os, sys
import numpy as np
import pyopencl as cl
from clutil import roundUp, createProgram, compareFormat

LWORKGROUP = (16, 16)

class Colorize(ChainedFilter):
	def __init__(self, canvas, range, hues, formatIn, formatOut=None):
		ChainedFilter.__init__(self, canvas)

		self.range = range
		self.hues = hues

		self.formatOut = formatOut

		options = []

		filename = os.path.join(os.path.dirname(__file__), 'colorize.cl')
		program = createProgram(canvas.context, canvas.devices, options, filename)

		if compareFormat(formatIn, (cl.Buffer, np.int32)) and \
			compareFormat(formatOut, (cl.Image, cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT8))):
			self.args = [
				np.array(self.range, np.int32),
				np.array(self.hues, np.int32),
				None,
				None,
				None,
			]

			self.kern = cl.Kernel(program, 'colorize_i32')
		elif compareFormat(formatIn, (cl.Buffer, np.float32)) and\
		   compareFormat(formatOut, (cl.Image, cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT8))):
			self.args = [
				np.array(self.range, np.float32),
				np.array(self.hues, np.int32),
				None,
				None,
				None,
			]

			self.kern = cl.Kernel(program, 'colorize_f32')
		else:
			raise NotImplemented()

		self.canvas = canvas

	def execute(self, input):
		if self.formatOut == None:
			output = input
		elif self.formatOut[0] == cl.Image:
			#check size of input does not exceed self.output
			output = cl.Image(self.canvas.context,
				cl.mem_flags.READ_WRITE,
				self.formatOut[1],
				input.shape
			)
		else:
			raise NotImplemented()

		self.args[2:5] = [
			input,
			np.array(input.shape, np.int32),
			output
		]

		gw = roundUp(input.shape, LWORKGROUP)

		self.kern(self.canvas.queue, gw, LWORKGROUP, *self.args)

		return output
