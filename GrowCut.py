__author__ = 'Marc de Klerk'

import pyopencl as cl
import numpy as np
import os
from clutil import roundUp, padArray2D, createProgram, formatForCLImage2D

LWORKGROUP = (16, 16)

cm = cl.mem_flags
szFloat = np.dtype(np.float32).itemsize
szInt = np.dtype(np.int32).itemsize

class GrowCut():
	lw = LWORKGROUP

	VON_NEUMANN = 0
	MOORE = 1

	def __init__(self, context, devices, img, neighbourhood=VON_NEUMANN, thresholdCanAttack=None, thresholdOverpowered=None):
		self.context = context

		if thresholdCanAttack == None:
			thresholdCanAttack = 6
		if thresholdOverpowered == None:
			thresholdOverpowered = 6

		options = [
			'-D CAN_ATTACK_THRESHOLD='+str(thresholdCanAttack),
			'-D OVER_PROWER_THRESHOLD='+str(thresholdOverpowered)
		]

		filename = os.path.join(os.path.dirname(__file__), 'growcut.cl')
		program = createProgram(context, devices, options, filename)

		self.kernCountEnemies = cl.Kernel(program, 'countEnemies')
		if neighbourhood == GrowCut.VON_NEUMANN:
			self.kernEvolve = cl.Kernel(program, 'evolveVonNeumann')
		elif neighbourhood == GrowCut.MOORE:
			self.kernEvolve = cl.Kernel(program, 'evolveMoore')

		if type(img) == cl.GLBuffer:
				raise ValueError('CL Buffer')
		elif type(img) == np.ndarray:
			raise NotImplementedError('NP arrays')
		elif type(img) == cl.GLTexture:
			raise NotImplementedError('GL Texture')
		elif type(img) == cl.Image:
			self.dImg = img

			width = img.get_image_info(cl.image_info.WIDTH)
			height = img.get_image_info(cl.image_info.HEIGHT)

			shapeNP = (height, width)
			shapeCL = (width, height)

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

#		elapsed = 0;
		event = self.kernCountEnemies(queue, self.gWorksize, self.lw, *argsCount)
		event.wait()
#		elapsed += event.profile.end - event.profile.start
#		print 'Execution time of test: {0} ms'.format(1e-6*elapsed)

		self.args[0] = self.dLabelsIn
		self.args[1] = self.dLabelsOut
		self.args[2] = self.dStrengthIn
		self.args[3] = self.dStrengthOut

#		elapsed = 0;
		event = self.kernEvolve(queue, self.gWorksize, self.lw, *self.args)
		event.wait()
#		elapsed += event.profile.end - event.profile.start
#		print 'Execution time of test: {0} ms'.format(1e-6*elapsed)

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
	from CLWindow import CLWindow
	from CLCanvas import CmLCanvas, Filter
	from Brush import Brush

	img = Image.open("/Users/marcdeklerk/msc/code/dataset/processed/source/800x600/GT04.png")
	if img.mode != 'RGBA':
		img = img.convert('RGBA')

	app = QtGui.QApplication(sys.argv)
	canvas = GLCanvas(img.size)
	window = GLWindow(canvas)

	clContext = canvas.clContext
	devices = clContext.get_info(cl.context_info.DEVICES)
	queue = cl.CommandQueue(clContext, properties=cl.command_queue_properties.PROFILING_ENABLE)

	shapeNP = (img.size[1], img.size[0])
	shapeNP = roundUp(shapeNP, GrowCut.lw)
	shapeCL = (shapeNP[1], shapeNP[0])

	hImg = padArray2D(np.array(img).view(np.uint32).squeeze(), shapeNP, 'edge')

	dImg = cl.Image(clContext,
		cl.mem_flags.READ_ONLY,
		cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8),
		shapeCL
	)
	cl.enqueue_copy(queue, dImg, hImg, origin=(0,0), region=shapeCL)

	dStrokes = cl.Buffer(clContext, cm.READ_WRITE, szInt*int(np.prod(shapeCL)))

	brushArgs = [
#		'__write_only image2d_t strokes',
		'__global int* strokes',
		'__global int* labels_in',
		'__global float* strength_in',
		'int label',
		'int canvasW'
	]
#	brushCode = 'write_imagef(strokes, gcoord, rgba2f4(label)/255.0f);\n'
	brushCode = 'strokes[gcoord.y*canvasW + gcoord.x] = label;\n'
	brushCode += 'strength_in[gcoord.y*canvasW + gcoord.x] = 1;\n'
	brushCode += 'labels_in[gcoord.y*canvasW + gcoord.x] = label;\n'

	brush = Brush(clContext, devices, brushArgs, brushCode)

	growCut = GrowCut(clContext, devices, dImg, GrowCut.VON_NEUMANN, 4, 4)

	label = 1

	iteration = 0
	refresh = 1

	def next():
		global iteration

		growCut.evolve(queue)

		if iteration % refresh == 0:
			window.updateCanvas()

			iteration += 1

	def mouseDrag(pos1, pos2):
		if pos1 == pos2:
			return

		brush.draw_gpu(queue, [dStrokes, growCut.dLabelsIn, growCut.dStrengthIn, np.int32(label), np.int32(shapeNP[1])], pos1, pos2)

		window.updateCanvas()

	def mousePress(pos):
		brush.draw_gpu(queue, [dStrokes, growCut.dLabelsIn, growCut.dStrengthIn, np.int32(label), np.int32(shapeNP[1])], pos)

		window.updateCanvas()

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

	#setup window
	filter = Filter((0, 3), (0, 240))
	window.addLayer('strokes', dStrokes, shapeCL, 0.25, np.int32, filter=filter)
	window.addLayer('labels', growCut.dLabelsOut, shapeCL, 0.5, np.int32, filter=filter)

	window.addLayer('image', dImg)

	filter = Filter((0, 9),(0, 240))
	window.addLayer('enemies', growCut.dEnemies, shapeCL, datatype=np.int32, filter=filter)

	filter = Filter((0, 1.0), (0, 240))
	window.addLayer('strength', growCut.dStrengthIn, shapeCL, 1.0, np.float32, filter=filter)

	window.addButton("start", functools.partial(timer.start, 0))
	window.addButton('next', next)
	window.setMousePress(mousePress)
	window.setMouseDrag(mouseDrag)
	window.setKeyPress(keyPress)

	window.show()
	sys.exit(app.exec_())