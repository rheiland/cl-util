__author__ = 'Marc de Klerk'

import pyopencl as cl
import numpy as np
import os
from clutil import roundUp, padArray2D, createProgram, isPow2

NEIGHBOURHOOD_VON_NEUMANN = 0
NEIGHBOURHOOD_MOORE = 1

LWORKGROUP = (16, 16)

cm = cl.mem_flags
szFloat = np.dtype(np.float32).itemsize
szInt = np.dtype(np.int32).itemsize

class GrowCut():
	lw = LWORKGROUP

	def __init__(self, context, devices, img, imgW=None, imgH=None):
		self.context = context

		filename = os.path.join(os.path.dirname(__file__), 'growcut.cl')
		program = createProgram(context, devices, [], filename)

		self.kernEvolve = cl.Kernel(program, 'evolve')

		if type(img) == np.ndarray:
			imgW = img.size[0]
			imgH = img.size[1]
			shapeNP = (img.size[1], img.size[0])

			hImg = padArray2D(np.array(img).view(np.uint32).squeeze(), roundUp(shapeNP, self.lw), 'edge')

			shapeCL = (hImg.shape[1], hImg.shape[0])
			shapeNP = hImg.shape

			self.dImg = cl.Buffer(context, cm.READ_ONLY | cm.COPY_HOST_PTR, hostbuf=hImg)
		else:
			if imgW == None or imgH == None:
				raise ValueError("CL Buffer width or height not provided")

			shapeCL = (imgW, imgH)
			shapeNP = (imgH, imgW)

			self.dImg = img

		self.hLabelsIn = np.zeros(shapeNP, np.int32)
		self.hLabelsOut = np.empty(shapeNP, np.int32)
		self.hStrengthIn = np.zeros(shapeNP, np.float32)
		self.hStrengthOut = np.empty(shapeNP, np.float32)
		self.hHasConverged = np.empty((1,), np.int32)

		self.hHasConverged[0] = False

		self.dLabelsIn = cl.Buffer(context, cm.READ_ONLY | cm.COPY_HOST_PTR, hostbuf=self.hLabelsIn)
		self.dLabelsOut = cl.Buffer(context, cm.READ_ONLY | cm.COPY_HOST_PTR, hostbuf=self.hLabelsOut)
		self.dStrengthIn = cl.Buffer(context, cm.READ_ONLY | cm.COPY_HOST_PTR, hostbuf=self.hStrengthIn)
		self.dStrengthOut = cl.Buffer(context, cm.READ_ONLY | cm.COPY_HOST_PTR, hostbuf=self.hStrengthOut)
		self.dHasConverged = cl.Buffer(context, cm.READ_ONLY | cm.COPY_HOST_PTR, hostbuf=self.hHasConverged)

		self.args = [
			self.dLabelsIn,
			self.dLabelsOut,
			self.dStrengthIn,
			self.dStrengthOut,
			self.dImg,
			self.dHasConverged,
			cl.LocalMemory(szInt*(self.lw[0]+2)*(self.lw[1]+2)),
			cl.LocalMemory(szInt*(self.lw[0]+2)*(self.lw[1]+2)),
			cl.LocalMemory(szInt*(self.lw[0]+2)*(self.lw[1]+2)),
			np.int32(imgW),
			np.int32(imgH)
		]

		self.gWorksize = roundUp(shapeCL, self.lw)

	def evolve(self, queue):
		self.args[0] = self.dLabelsIn
		self.args[1] = self.dLabelsOut
		self.args[2] = self.dStrengthIn
		self.args[3] = self.dStrengthOut

		self.kernEvolve(queue, self.gWorksize, self.lw, *self.args).wait()

		cl.enqueue_copy(queue, self.hLabelsOut, self.dLabelsOut).wait()
		cl.enqueue_copy(queue, self.hStrengthOut, self.dStrengthOut).wait()

		dTmp = self.dLabelsOut
		self.dLabelsOut = self.dLabelsIn
		self.dLabelsIn = dTmp

		dTmp = self.dStrengthOut
		self.dStrengthOut = self.dStrengthIn
		self.dStrengthIn = dTmp

if __name__ == "__main__":
	import functools
	import Image
	import sys
	from PyQt4 import QtCore, QtGui, QtOpenGL
	from GLWindow import GLWindow
	from Colorize import Colorize

	img = Image.open("/Users/marcdeklerk/msc/code/dataset/processed/source/800x600/GT04.png")
	if img.mode != 'RGBA':
		img = img.convert('RGBA')

	app = QtGui.QApplication(sys.argv)
	window = GLWindow((img.size[1], img.size[0]))
	context = window.context
	devices = context.get_info(cl.context_info.DEVICES)
	queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

	colorize = Colorize(context, devices)

	shapeNP = (img.size[1], img.size[0])
	shapeNP = roundUp(shapeNP, GrowCut.lw)
	hImg = padArray2D(np.array(img).view(np.uint32).squeeze(), shapeNP, 'edge')

	vLabels = window.addView(hImg.shape, 'Labels')
	vImg = window.addViewNp(hImg, 'Image')

	growCut = GrowCut(context, devices, vImg, shapeNP[1], shapeNP[0])

	reversedHue = (240, 0)
	def mapLabels():
		m = 0
		M = 2
		colorize.colorize(queue, growCut.dLabelsIn, val=(m, M), hue=reversedHue, dOut=vLabels, typeIn=np.int32)

	window.setMap(vLabels, mapLabels)

	label = 1

	def intercept():
		global label
		label = not label
		pass

	def next():
		growCut.evolve(queue)
		window.updateCanvas()

	def mousePress(pos):
		cl.enqueue_copy(queue, growCut.hLabelsIn, growCut.dLabelsIn).wait()
		cl.enqueue_copy(queue, growCut.hStrengthIn, growCut.dStrengthIn).wait()

		growCut.hLabelsIn[pos[1], pos[0]] = label
		growCut.hStrengthIn[pos[1], pos[0]] = 1

		cl.enqueue_copy(queue, growCut.dLabelsIn, growCut.hLabelsIn).wait()
		cl.enqueue_copy(queue, growCut.dStrengthIn, growCut.hStrengthIn).wait()

	def keyPress(key):
		global label

		if key   == QtCore.Qt.Key_0: label = 0
		elif key == QtCore.Qt.Key_1: label = 1
		elif key == QtCore.Qt.Key_2: label = 2
		elif key == QtCore.Qt.Key_3: label = 3

	timer = QtCore.QTimer()
	timer.timeout.connect(next)

	window.addButton("start", functools.partial(timer.start, 0))
	window.setMousePress(mousePress)
	window.setKeyPress(keyPress)

	window.show()
	sys.exit(app.exec_())
