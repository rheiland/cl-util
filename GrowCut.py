__author__ = 'Marc de Klerk'

import pyopencl as cl
import numpy as np
import os
from clutil import roundUp, padArray2D, createProgram

NEIGHBOURHOOD_VON_NEUMANN = 0
NEIGHBOURHOOD_MOORE = 0

LWORKGROUP = (16, 16)

cm = cl.mem_flags
szFloat = np.dtype(np.float32).itemsize
szInt = np.dtype(np.int32).itemsize

class GrowCut():
	lw = LWORKGROUP

	def __init__(self, context, devices, img, neighbourhood=NEIGHBOURHOOD_VON_NEUMANN):
		self.context = context

		options = [
			'-D NEIGHBOURHOOD='+str(neighbourhood)
		]

		filename = os.path.join(os.path.dirname(__file__), 'growcut.cl')
		program = createProgram(context, devices, options, filename)

		self.kernEvolve = cl.Kernel(program, 'evolve')
		self.kernCountEnemies = cl.Kernel(program, 'countEnemies')

		if type(img) == cl.GLBuffer:
				raise ValueError("CL Buffer not implemented")
		elif type(img) == np.ndarray:
			raise NotImplementedError("NP arrays not implemented")
		elif type(img) == cl.GLTexture:
			self.dImg = img

			width = img.get_image_info(cl.image_info.WIDTH)
			height = img.get_image_info(cl.image_info.HEIGHT)

			shapeNP = (height, width)
			shapeCL = (width, height)

		elif type(img) == cl.Image:
			raise NotImplementedError("CL image not implemented")

		self.hEnemiesIn = np.zeros(shapeNP,np.int32)
		self.hLabelsIn = np.zeros(shapeNP,np.int32)
		self.hLabelsOut = np.empty(shapeNP, np.int32)
		self.hStrengthIn = np.zeros(shapeNP, np.float32)
		self.hStrengthOut = np.zeros(shapeNP, np.float32)
		self.hHasConverged = np.empty((1,), np.int32)

		self.hHasConverged[0] = False

		self.dEnemies = cl.Buffer(context, cm.READ_ONLY | cm.COPY_HOST_PTR, hostbuf=self.hEnemiesIn)
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
			self.dEnemies,
			self.dHasConverged,
			cl.LocalMemory(szInt*(self.lw[0]+2)*(self.lw[1]+2)),
			cl.LocalMemory(szFloat*(self.lw[0]+2)*(self.lw[1]+2)),
			cl.LocalMemory(4*szFloat*(self.lw[0]+2)*(self.lw[1]+2)),
			cl.LocalMemory(szInt*(self.lw[0]+2)*(self.lw[1]+2)),
			self.dImg,
			cl.Sampler(clContext, False, cl.addressing_mode.NONE, cl.filter_mode.NEAREST)
		]

		self.gWorksize = roundUp(shapeCL, self.lw)

	def evolve(self, queue):
		argsCount = [
			self.dLabelsIn,
			cl.LocalMemory(szInt*(self.lw[0]+2)*(self.lw[1]+2)),
			self.dEnemies
		]

		elapsed = 0;
		event = self.kernCountEnemies(queue, self.gWorksize, self.lw, *argsCount)
		event.wait()
		elapsed += event.profile.end - event.profile.start
		print 'Execution time of test: {0} ms'.format(1e-6*elapsed)

		self.args[0] = self.dLabelsIn
		self.args[1] = self.dLabelsOut
		self.args[2] = self.dStrengthIn
		self.args[3] = self.dStrengthOut

		elapsed = 0;
		event = self.kernEvolve(queue, self.gWorksize, self.lw, *self.args)
		event.wait()
		elapsed += event.profile.end - event.profile.start

		print 'Execution time of test: {0} ms'.format(1e-6*elapsed)

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

	def mapEnemies():
		m = 0
		M = 4
		colorize.colorize(queue, growCut.dEnemies, val=(m, M), dOut=vEnemies, typeIn=np.int32)

	def mapLabels():
		m = 0
		M = 10
		colorize.colorize(queue, growCut.dLabelsIn, val=(m, M), dOut=vLabels, typeIn=np.int32)

	def mapStrength():
		m = 0
		M = 2
		colorize.colorize(queue, growCut.dStrengthIn, val=(m, M), dOut=vStrength)

	vEnemies = window.addView(shapeNP, 'enemies', cm.WRITE_ONLY, True)
	vStrokes = window.addView(shapeNP, 'strokes', cm.WRITE_ONLY, True)
	vStrength = window.addView(shapeNP, 'strength', cm.WRITE_ONLY, True)
	vLabels = window.addView(shapeNP, 'labels', cm.READ_WRITE, True)
	vImg = window.addViewNp(hImg, 'Image', cl.mem_flags.READ_ONLY)

	window.setLayerMap('enemies', mapEnemies)
	window.setLayerMap('labels', mapLabels)
	window.setLayerMap('strength', mapStrength)
	window.setLayerOpacity('strokes', 0.0)
	window.setLayerOpacity('strength', 0.0)
	window.setLayerOpacity('labels', 0.7)
	window.setLayerOpacity('strokes', 1.0)

	brushArgs = [
#		'__write_only image2d_t strokes',
		'__global uint* strokes',
		'__global int* labels_in',
		'__global float* strength_in',
		'int label',
		'int canvasW'
	]
#	brushCode = 'write_imagef(strokes, gcoord, rgba2f4(label)/255.0f);\n'
	brushCode = 'strokes[gcoord.y*canvasW + gcoord.x] = 0xFF000000 | 50*label;\n'
	brushCode += 'strength_in[gcoord.y*canvasW + gcoord.x] = 1;\n'
	brushCode += 'labels_in[gcoord.y*canvasW + gcoord.x] = label;\n'

	brush = Brush(clContext, devices, brushArgs, brushCode)

	growCut = GrowCut(clContext, devices, vImg)

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

		brush.draw_gpu(queue, [vStrokes, growCut.dLabelsIn, growCut.dStrengthIn, np.int32(label), np.int32(shapeNP[1])], pos1, pos2)

		if iteration % refresh == 0:
			window.updateCanvas()

		iteration += 1

	def mousePress(pos):
		global iteration

		brush.draw_gpu(queue, [vStrokes, growCut.dLabelsIn, growCut.dStrengthIn, np.int32(label), np.int32(shapeNP[1])], pos)

		if iteration % refresh == 0:
			window.updateCanvas()

		iteration += 1

	def keyPress(key):
		global label

		if key   == QtCore.Qt.Key_0: label = 0
		elif key == QtCore.Qt.Key_1: label = 1
		elif key == QtCore.Qt.Key_2: label = 2
		elif key == QtCore.Qt.Key_3: label = 3
		elif key == QtCore.Qt.Key_4: label = 4
		elif key == QtCore.Qt.Key_5: label = 5
		elif key == QtCore.Qt.Key_6: label = 6
		elif key == QtCore.Qt.Key_7: label = 7
		elif key == QtCore.Qt.Key_8: label = 8

	timer = QtCore.QTimer()
	timer.timeout.connect(next)

	window.addButton("start", functools.partial(timer.start, 0))
	window.addButton('next', next)
	window.setMousePress(mousePress)
	window.setMouseDrag(mouseDrag)
	window.setKeyPress(keyPress)

	window.show()
	sys.exit(app.exec_())