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

	def __init__(self, context, devices, img, shape=None):
		self.context = context

		filename = os.path.join(os.path.dirname(__file__), 'growcut.cl')
		program = createProgram(context, devices, [], filename)

		self.kernEvolve = cl.Kernel(program, 'evolve')

		if type(img) == cl.GLBuffer:
			if shape == None:
				raise ValueError("CL Buffer width or height not provided")

			shapeCL = (shape[1], shape[0])

			self.dImg = img
		elif type(img) == np.ndarraye:
			raise NotImplementedError("NP arrays not implemented")
		elif type(img) == cl.GLTexture:
			raise NotImplementedError("GL textures not implemented")
		elif type(img) == cl.Image:
			raise NotImplementedError("CL image not implemented")

		self.hLabelsIn = np.zeros(shape, np.int32)
		self.hLabelsOut = np.empty(shape, np.int32)
		self.hStrengthIn = np.zeros(shape, np.float32)
		self.hStrengthOut = np.empty(shape, np.float32)
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
			np.int32(shape[1]),
			np.int32(shape[0])
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
	from Brush import Brush

	img = Image.open("/Users/marcdeklerk/msc/code/dataset/processed/source/800x600/GT04.png")
	if img.mode != 'RGBA':
		img = img.convert('RGBA')

	app = QtGui.QApplication(sys.argv)
	window = GLWindow(img.size)
	clContext = window.clContext
	glContext = window.glContext
	devices = clContext.get_info(cl.context_info.DEVICES)
	queue = cl.CommandQueue(clContext, properties=cl.command_queue_properties.PROFILING_ENABLE)

	colorize = Colorize(clContext, devices)

	shapeNP = (img.size[1], img.size[0])
	shapeNP = roundUp(shapeNP, GrowCut.lw)
	hImg = padArray2D(np.array(img).view(np.uint32).squeeze(), shapeNP, 'edge')

	reversedHue = (240, 0)
	def mapLabels():
		m = 0
		M = 3
		colorize.colorize(queue, growCut.dLabelsIn, val=(m, M), hue=reversedHue, dOut=vLabels, typeIn=np.int32)

	vLabels = window.addView(shapeNP, 'labels', mapLabels)
	vImg = window.addViewNp(hImg, 'Image', buffer=True)

	window.setMap("labels", mapLabels)

	brushArgs = [
		#'__write_only image2d_t canvas',
		'__global int* labels_in',
		'__global float* strength_in',
		'int label',
		'int canvasW'
	]
	#brushCode = 'write_imagef(canvas, gcoord, rgba2f4(color)/255.0f);\n'
	brushCode = 'labels_in[gcoord.y*canvasW + gcoord.x] = label;\n'
	brushCode += 'strength_in[gcoord.y*canvasW + gcoord.x] = 1;\n'

	brush = Brush(clContext, devices, brushArgs, brushCode)

	growCut = GrowCut(clContext, devices, vImg, shapeNP)

	label = 1

	def intercept():
		global label
		label = not label
		pass

	def next():
		growCut.evolve(queue)
		window.updateCanvas()

	iteration = 0
	refresh = 50

	def mouseDrag(pos1, pos2):
		global iteration

		if pos1 == pos2:
			return

		brush.draw_gpu(queue, [growCut.dLabelsIn, growCut.dStrengthIn, np.int32(label), np.int32(shapeNP[1])], pos1, pos2)

		if iteration % refresh == 0:
			window.updateCanvas()

		iteration += 1

	def mousePress(pos):
		global iteration

		brush.draw_gpu(queue, [growCut.dLabelsIn, growCut.dStrengthIn, np.int32(label), np.int32(shapeNP[1])], pos)

		if iteration % refresh == 0:
			window.updateCanvas()

		iteration += 1

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
	window.setMouseDrag(mouseDrag)
	window.setKeyPress(keyPress)

	window.show()
	sys.exit(app.exec_())
