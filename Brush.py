from Tool import Tool
import Image
import ImageDraw
import numpy as np
import os

import pyopencl as cl
from clutil import roundUp

try:
	from OpenGL.GL import *
except ImportError:
	raise ImportError("Error importing PyOpenGL")

cm = cl.mem_flags
szFloat = np.dtype(np.float32).itemsize
szInt = np.dtype(np.int32).itemsize

LWORKGROUP = (16, 16)

def argb2abgr(c):
	a = (c >> 24) & 0xFF
	r = (c >> 16) & 0xFF
	g = (c >> 8 ) & 0xFF
	b = (c      ) & 0xFF

	return a << 24 | b << 16 | g << 8 | r

class Brush(Tool):
	DEFAULT_COLOR = argb2abgr(0xFFFF00FF)
	lw = LWORKGROUP

	def __init__(self, clContext, devices, arguments, code):
		Tool.__init__(self)

		self.radius = 10
		self.color = Brush.DEFAULT_COLOR

		filename = os.path.join(os.path.dirname(__file__), 'brush.cl')

		kernelFile = open(filename, 'r')
		kernelStr = kernelFile.read()
		kernelStr = kernelStr.replace('OUTPUT_ARGS', ',\n'+',\n'.join(arguments))
		kernelStr = kernelStr.replace('CODE', code)

		program = cl.Program(clContext, kernelStr)
		program.build([], devices)

		self.kernStroke = cl.Kernel(program, 'stroke')
		self.kernPoint = cl.Kernel(program, 'point')

	def getCursor(self):
		shape = (self.diameter, self.diameter)
		cursor = Image.new('RGBA', shape)
		draw = ImageDraw.Draw(cursor)

		color = self.colorCursor[self.type]

		draw.ellipse((0, 0, self.diameter-1, self.diameter-1), fill=color)

		return cursor

	def setType(self, type):
		self.type = type;

	@property
	def diameter(self):
		return 2*self.radius + 1

	def getRadius(self):
		return self.radius

	def setRadius(self, radius, increase=None):
		if increase == '+':
			self.radius += radius
		elif increase == '-':
			self.radius -= radius
		else:
			self.radius = radius

		if radius <= 0:
			self.radius = 1

	def draw_gpu(self, queue, arguments, pos1, pos2=None, shape=None):
		if pos2 == None:
			argsPoint = [
				np.array(pos1, np.int32),
				np.int32(self.radius),
				np.uint32(self.color),
			]
			argsPoint += arguments

			gw = roundUp((self.diameter, self.diameter), self.lw)

			self.kernPoint(queue, gw, self.lw, *argsPoint).wait()
		else:
			if pos2[1] < pos1[1]:
				tmpPos = pos1
				pos1 = pos2
				pos2 = tmpPos

			dx = pos2[0]-pos1[0]
			dy = pos2[1]-pos1[1]

			if dy == 0:
				return

			row = self.radius * (dy / np.sqrt(dx**2 + dy**2))

			argsStroke = [
				(np.float32(dx)/np.float32(dy) if dy != 0 else np.finfo(np.float32).max),
				np.array([pos1[0]-abs(row), pos1[1]], np.int32),
				np.int32(dy),
				np.int32(2*abs(row) + 1),
				np.uint32(self.color),
			]
			argsStroke += arguments

			gw = roundUp((int(2*abs(row) + 1), abs(dy)), self.lw)

			self.kernStroke(queue, gw, self.lw, *argsStroke).wait()
