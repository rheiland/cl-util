__author__ = 'marcdeklerk'

import pyopencl as cl
import numpy as np
import os
from clutil import roundUp, padArray2D, createProgram, Buffer2D

LWORKGROUP = (16, 16)

cm = cl.mem_flags
szFloat = np.dtype(np.float32).itemsize
szInt = np.dtype(np.int32).itemsize

class SharedMatting():
	lw = LWORKGROUP

	def __init__(self, context, devices, dImg):
		self.context = context
		self.queue = cl.CommandQueue(context)

		self.dim = dImg.dim

		options = [
			'-D IMAGEW='+str(self.dim[0]),
			'-D IMAGEH='+str(self.dim[1]),
		]

		self.dImg = dImg
		self.dFg = Buffer2D(context, cl.mem_flags.READ_WRITE, self.dim, np.uint32)
		self.dBg = Buffer2D(context, cl.mem_flags.READ_WRITE, self.dim, np.uint32)
		self.dLcv = Buffer2D(context, cl.mem_flags.READ_WRITE, self.dim, np.float32)
		self.dAlpha = Buffer2D(context, cl.mem_flags.READ_WRITE, self.dim, np.float32)

		filename = os.path.join(os.path.dirname(__file__), 'sharedmatting.cl')
		program = createProgram(context, devices, options, filename)

		self.kernGather = cl.Kernel(program, 'gather')
		self.kernLcv = cl.Kernel(program, 'local_color_variation')
		self.kernRefine = cl.Kernel(program, 'refine')

	def calcMatte(self, dTri):
		gWorksize = roundUp(self.dim, SharedMatting.lw)

		args = [
			self.dImg,
			self.dLcv,
		]
		self.kernLcv(self.queue, gWorksize, SharedMatting.lw, *args)

		args = [
			self.dImg,
			dTri,
			self.dFg,
			self.dBg,
			]
		self.kernGather(self.queue, gWorksize, SharedMatting.lw, *args)

		args = [
			self.dImg,
			dTri,
			self.dFg,
			self.dBg,
			self.dAlpha,
			self.dLcv,
			]
		self.kernRefine(self.queue, gWorksize, SharedMatting.lw, *args)

		self.queue.finish()

if __name__ == "__main__":
	import Image
	from PyQt4 import QtGui
	from CLWindow import CLWindow
	from CLCanvas import CLCanvas
	from Colorize import Colorize
	import sys

	img = Image.open("/Users/marcdeklerk/msc/code/dataset/processed/source/800x600/GT04.png").convert('RGBA')
	tri = Image.open("/Users/marcdeklerk/msc/code/dataset/processed/trimap1/800x600/GT04.png").convert('RGBA')

	app = QtGui.QApplication(sys.argv)
	canvas = CLCanvas(img.size)
	window = CLWindow(canvas)

	context = canvas.context
	devices = context.get_info(cl.context_info.DEVICES)
	queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

	shape = (img.size[1], img.size[0])
	shape = roundUp(shape, SharedMatting.lw)
	dim = (shape[1], shape[0])

	hImg = padArray2D(np.array(img).view(np.uint32).squeeze(), shape, 'edge')
	hTri = padArray2D(np.array(tri).view(np.uint32).squeeze(), shape, 'edge')

	dImg = Buffer2D(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hImg)
	dTri = Buffer2D(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hTri)

	sm = SharedMatting(context, devices, dImg)

	sm.calcMatte(dTri)

	colorize = Colorize(canvas)

	#setup window
	window.addLayer('tri', dTri, 0.0)

	filter = colorize.factory((Buffer2D, np.float32), (0, 1.0), (0, 0), (0, 0), (0, 1))
	window.addLayer('alpha', sm.dAlpha, 1.0, filter=filter)

	window.addLayer('fg', sm.dFg)
	window.addLayer('bg', sm.dBg)

	filter = colorize.factory((Buffer2D, np.float32), (0, 5000), hues=Colorize.HUES.REVERSED)
	window.addLayer('lcv', sm.dLcv, 1.0, filter=filter)

	window.addLayer('image', dImg)

	window.resize(1000, 700)
	window.move(2000, 0)
	window.show()
	sys.exit(app.exec_())