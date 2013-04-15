__author__ = 'Marc de Klerk'

import pyopencl as cl
import numpy as np
import os
from clutil import roundUp, padArray2D, createProgram, Buffer2D
from StreamCompact import StreamCompact, TileList

LWORKGROUP = (16, 16)

cm = cl.mem_flags
szFloat = np.dtype(np.float32).itemsize
szInt = np.dtype(np.int32).itemsize

TILEW = 16
TILEH = 16

WEIGHT = '(1.0f - X/1.7320508075688772f)'

class GrowCut():
	lw = LWORKGROUP

	VON_NEUMANN = 0
	MOORE = 1

	lWorksizeTiles16 = (16, 16)

	WEIGHT_DEFAULT  = '(1.0f-X/1.7320508075688772f)'
	WEIGHT_POW2     = '(1.0f-pown(X/1.7320508075688772f,2))'
	WEIGHT_POW3     = '(1.0f-pown(X/1.7320508075688772f,3))'
	WEIGHT_POW1_5   = '(1.0f-pow(X/1.7320508075688772f,1.5))'
	WEIGHT_POW_SQRT = '(1.0f-sqrt(X/1.7320508075688772f))'

	CAN_ATTACK_THRESHOLD_DEAFULT = 6
	OVER_POWER_THRESHOLD_DEFAULT = 6

	def __init__(self, context, devices, img, neighbourhood=VON_NEUMANN, thresholdCanAttack=None, thresholdOverpowered=None, weight=None):
		self.context = context

		if thresholdCanAttack == None:
			thresholdCanAttack = GrowCut.CAN_ATTACK_THRESHOLD_DEAFULT
		if thresholdOverpowered == None:
			thresholdOverpowered = GrowCut.OVER_PROWER_THRESHOLD_DEFAULT
		if weight == None:
			weight = GrowCut.WEIGHT_DEFAULT

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
			dim = (width, height)

		tilesW = dim[0]/TILEW
		tilesH = dim[1]/TILEH

		streamCompact = StreamCompact(context, devices, tilesW*tilesH)

		shapeTiles = (tilesW, tilesH)

		self.tilelist = TileList(context, devices, shapeTiles)

		self.hEnemiesIn = np.zeros(shapeNP,np.int32)
		self.hLabelsIn = np.zeros(shapeNP,np.int32)
		self.hLabelsOut = np.empty(shapeNP, np.int32)
		self.hStrengthIn = np.zeros(shapeNP, np.float32)
		self.hStrengthOut = np.zeros(shapeNP, np.float32)
		self.hHasConverged = np.empty((1,), np.int32)

		self.hHasConverged[0] = False

		self.dEnemies = Buffer2D(context, cm.READ_ONLY | cm.COPY_HOST_PTR, hostbuf=self.hEnemiesIn)
		self.dLabelsIn = Buffer2D(context, cm.READ_ONLY | cm.COPY_HOST_PTR, hostbuf=self.hLabelsIn)
		self.dLabelsOut = Buffer2D(context, cm.READ_ONLY | cm.COPY_HOST_PTR, hostbuf=self.hLabelsOut)
		self.dStrengthIn = Buffer2D(context, cm.READ_ONLY | cm.COPY_HOST_PTR, hostbuf=self.hStrengthIn)
		self.dStrengthOut = Buffer2D(context, cm.READ_ONLY | cm.COPY_HOST_PTR, hostbuf=self.hStrengthOut)
		self.dHasConverged = cl.Buffer(context, cm.READ_ONLY | cm.COPY_HOST_PTR, hostbuf=self.hHasConverged)

		self.args = [
			self.tilelist.dList,
			self.dLabelsIn,
			self.dLabelsOut,
			self.dStrengthIn,
			self.dStrengthOut,
			self.dEnemies,
			self.dHasConverged,
			np.int32(self.tilelist.iteration),
			self.tilelist.dTiles,
			cl.LocalMemory(szInt*9),
			cl.LocalMemory(szInt*(TILEW+2)*(TILEH+2)),
			cl.LocalMemory(szFloat*(TILEW+2)*(TILEH+2)),
			cl.LocalMemory(4*szFloat*(TILEW+2)*(TILEH+2)),
			cl.LocalMemory(szInt*(TILEW+2)*(TILEH+2)),
			self.dImg,
			cl.Sampler(clContext, False, cl.addressing_mode.NONE, cl.filter_mode.NEAREST)
		]

		self.gWorksize = roundUp(shapeCL, self.lw)
		self.gWorksizeTiles16 = roundUp(dim, self.lWorksizeTiles16)

		options = [
			'-D CAN_ATTACK_THRESHOLD='+str(thresholdCanAttack),
			'-D OVER_PROWER_THRESHOLD='+str(thresholdOverpowered),
			'-D TILESW='+str(tilesW),
			'-D TILESH='+str(tilesH),
			'-D IMAGEW='+str(dim[0]),
			'-D IMAGEH='+str(dim[1]),
			'-D TILEW='+str(TILEW),
			'-D TILEH='+str(TILEH),
			'-D G_NORM(X)='+weight
		]

		filename = os.path.join(os.path.dirname(__file__), 'growcut.cl')
		program = createProgram(context, devices, options, filename)

		self.kernCountEnemies = cl.Kernel(program, 'countEnemies')
		if neighbourhood == GrowCut.VON_NEUMANN:
			self.kernEvolve = cl.Kernel(program, 'evolveVonNeumann')
		elif neighbourhood == GrowCut.MOORE:
			self.kernEvolve = cl.Kernel(program, 'evolveMoore')

		self.isComplete = False

	def evolve(self, queue):
		self.isComplete = False

		self.tilelist.flag()

		if self.tilelist.length == 0:
			self.isComplete = True
			return

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

		self.tilelist.increment()

		self.args[1] = self.dLabelsIn
		self.args[2] = self.dLabelsOut
		self.args[3] = self.dStrengthIn
		self.args[4] = self.dStrengthOut

		self.args[7] = np.int32(self.tilelist.iteration)

#		elapsed = 0;
		gWorksize = (TILEW*self.tilelist.length, TILEH)
		lWorksize = self.lWorksizeTiles16
#		gWorksize = self.gWorksize
#		lWorksize = self.lw
		event = self.kernEvolve(queue, gWorksize, lWorksize, *self.args)
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
	from CLCanvas import CLCanvas, Filter
	from Brush import Brush
	from Colorize import Colorize

	img = Image.open("/Users/marcdeklerk/msc/code/dataset/processed/source/800x600/GT04.png")
	if img.mode != 'RGBA':
		img = img.convert('RGBA')

	app = QtGui.QApplication(sys.argv)
	canvas = CLCanvas(img.size)
	window = CLWindow(canvas)

	clContext = canvas.context
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

	dStrokes = Buffer2D(clContext, cm.READ_WRITE, shapeCL, dtype=np.int32)

	growCut = GrowCut(clContext, devices, dImg, GrowCut.MOORE, 6, 6, GrowCut.WEIGHT_POW3)

	brushArgs = [
#		'__write_only image2d_t strokes',
		'__global int* strokes',
		'__global int* labels_in',
		'__global float* strength_in',
		'__global int* tiles',
		'int iteration',
		'int label',
		'int canvasW'
	]
#	brushCode = 'write_imagef(strokes, gcoord, rgba2f4(label)/255.0f);\n'
	brushCode = 'strokes[gcoord.y*canvasW + gcoord.x] = label;\n'
	brushCode += 'strength_in[gcoord.y*canvasW + gcoord.x] = 1;\n'
	brushCode += 'labels_in[gcoord.y*canvasW + gcoord.x] = label;\n'
	brushCode += 'tiles[(gcoord.y/{0})*{1} + gcoord.x/{2}] = iteration;\n'.format(TILEH, growCut.tilelist.shape[0], TILEW)

	brush = Brush(clContext, devices, brushArgs, brushCode)

	label = 1

	iteration = 0
	refresh = 100

	def next():
		global iteration

		growCut.evolve(queue)

		if growCut.isComplete:
			print 'complete'
			timer.stop()

			window.updateCanvas()

		if iteration % refresh == 0:
			window.updateCanvas()

		iteration += 1

	def mouseDrag(pos1, pos2):
		if pos1 == pos2:
			return

		timer.stop()
		brush.draw_gpu(queue, [dStrokes, growCut.dLabelsIn, growCut.dStrengthIn, growCut.tilelist.dTiles, np.int32(growCut.tilelist.iteration), np.int32(label), np.int32(shapeNP[1])], pos1, pos2)
		queue.finish()
		timer.start()

		window.updateCanvas()

	def mousePress(pos):
		timer.stop()
		brush.draw_gpu(queue, [dStrokes, growCut.dLabelsIn, growCut.dStrengthIn, growCut.tilelist.dTiles, np.int32(growCut.tilelist.iteration), np.int32(label), np.int32(shapeNP[1])], pos)
		queue.finish()
		timer.start()

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

	colorize = Colorize(canvas)

	#setup window
	filter = colorize.factory((Buffer2D, np.int32), (0, 9))
	window.addLayer('strokes', dStrokes, 0.25, filter=filter)
	window.addLayer('labels', growCut.dLabelsOut, 0.5, filter=filter)
	window.addLayer('image', dImg)

	filter = colorize.factory((Buffer2D, np.int32), (0, 4))
	window.addLayer('enemies', growCut.dEnemies, filter=filter)

	filter = colorize.factory((Buffer2D, np.float32), (0, 1.0), hues=Colorize.HUES.REVERSED)
	window.addLayer('strength', growCut.dStrengthIn, 1.0, filter=filter)

	options = [
		'-D IMAGEW={0}'.format(shapeCL[0]),
		'-D IMAGEH={0}'.format(shapeCL[1]),
		'-D TILESW='+str(growCut.tilelist.shape[0]),
		'-D TILESH='+str(growCut.tilelist.shape[1])
	]

	filename = os.path.join(os.path.dirname(__file__), 'graphcut_filter.cl')
	program = createProgram(canvas.context, canvas.devices, options, filename)

	kernTileList = cl.Kernel(program, 'tilelist_growcut')

	class TileListFilter():
		class Filter(Colorize.Filter):
			def __init__(self, format, hues, tileflags):
				Colorize.Filter.__init__(self, kernTileList, format, (0, tileflags.iteration), hues)

				self.tileflags = tileflags

			def execute(self, queue, args):
				range = np.array([self.tileflags.iteration-1, self.tileflags.iteration], np.int32)

				kernTileList(queue, growCut.gWorksizeTiles16, growCut.lWorksizeTiles16, range, self.hues, *args)

		def __init__(self, canvas):
			pass

		def factory(self, tileflags, hues=None):
			if hues == None:
				hues = Colorize.HUES.STANDARD

			return TileListFilter.Filter((Buffer2D, np.int32), hues, tileflags)

	tilelistfilter = TileListFilter(canvas)

	filter = tilelistfilter.factory(growCut.tilelist, hues=Colorize.HUES.REVERSED)
	window.addLayer('tiles', growCut.tilelist.dTiles, filter=filter)

	window.addButton("start", functools.partial(timer.start, 0))
	window.addButton('next', next)
	window.setMousePress(mousePress)
	window.setMouseDrag(mouseDrag)
	window.setKeyPress(keyPress)

#	growCut.tilelist.flag(StreamCompact.OPERATOR_GTE, -1)

#	timer.start()

	window.resize(1000, 700)
	window.show()
	sys.exit(app.exec_())