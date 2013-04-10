__author__ = 'marcdeklerk'

import os
import numpy as np
import Image
import pyopencl as cl
import msclib, msclib.clutil as clutil
from msclib import context, queue, device
from StreamCompact import StreamCompact, TileList
from clutil import createProgram, cl, roundUp, ceil_divi, padArray2D, Buffer2D
import functools

from PyQt4 import QtCore, QtGui, QtOpenGL
from CLWindow import CLWindow
from CLCanvas import CLCanvas
from Colorize import Colorize
import sys
import time

szFloat = np.dtype(np.float32).itemsize
szInt = np.dtype(np.int32).itemsize
szChar = np.dtype(np.uint8).itemsize
sz32 = 4

cm = cl.mem_flags

WAVE_BREDTH = 32
WAVE_LENGTH = 8
WAVES_PER_WORKGROUP = 4

TILEW = WAVE_BREDTH
TILEH = (WAVES_PER_WORKGROUP*WAVE_LENGTH)

LAMBDA = 60
EPSILON = 0.05

lWorksizeTiles16 = (16, 16)

class GraphCut:
	lWorksizeTiles16 = (16, 16)
	lWorksizeSingleWave = (WAVE_BREDTH, 1)
	lWorksizeWaves = (WAVE_BREDTH, WAVES_PER_WORKGROUP)
	lWorksizeBorderAdd = (WAVE_BREDTH, )

	def __init__(self, context, devices, dImg, lamda=LAMBDA):
		MAX_HEIGHT = height*width

		options = []
		options += ['-D MAX_HEIGHT='+repr(MAX_HEIGHT)]
		options += ['-D LAMBDA='+repr(LAMBDA)]
		options += ['-D EPSILON='+repr(EPSILON)]
		options += ['-D TILESW='+str(tilesW)]
		options += ['-D TILESH='+str(tilesH)]

		context = context
		devices = context.get_info(cl.context_info.DEVICES)
		self.queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

		dir = os.path.dirname(__file__)

		program = clutil.createProgram(context, devices, options, os.path.join(dir, 'graphcut.cl'))
		self.kernInitGC = cl.Kernel(program, 'init_gc')
		self.kernLoadTiles = cl.Kernel(program, 'load_tiles')
		self.kernPushUp = cl.Kernel(program, 'push_up')
		self.kernPushDown = cl.Kernel(program, 'push_down')
		self.kernPushLeft = cl.Kernel(program, 'push_left')
		self.kernPushRight = cl.Kernel(program, 'push_right')
		self.kernRelabel = cl.Kernel(program, 'relabel')
		self.kernAddBorder = cl.Kernel(program, 'add_border')
		self.kernInitBfs = cl.Kernel(program, 'init_bfs')
		self.kernBfsIntraTile = cl.Kernel(program, 'bfs_intratile')
		self.kernBfsInterTile = cl.Kernel(program, 'bfs_intertile')
		self.kernCheckCompletion = cl.Kernel(program, 'check_completion')

		dim = dImg.dim

		size = sz32*dim[0]*dim[1]
		sizeSingleWave = sz32*dim[0]*(dim[1]/TILEH)
		sizeCompressedWave = sz32*dim[0]*(dim[1]/TILEH)

		self.dImg = dImg
		self.dUp = Buffer2D(context, cm.READ_WRITE | cm.COPY_HOST_PTR, size)
		self.dDown = Buffer2D(context, cm.READ_WRITE | cm.COPY_HOST_PTR, size)
		self.dLeft = Buffer2D(context, cm.READ_WRITE | cm.COPY_HOST_PTR, size)
		self.dRight = Buffer2D(context, cm.READ_WRITE | cm.COPY_HOST_PTR, size)
		self.dHeight = Buffer2D(context, cm.READ_WRITE | cm.COPY_HOST_PTR, size)
		self.dHeight2 = Buffer2D(context, cm.READ_WRITE | cm.COPY_HOST_PTR, size)
		self.dBorder = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, sizeSingleWave)
		self.dCanUp = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, sizeSingleWave)
		self.dCanDown = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, sizeSingleWave)
		self.dCanLeft = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, sizeSingleWave)
		self.dCanRight = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, sizeSingleWave)
		self.dIsCompleted = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, szInt)

		tilesW = dim[0]/TILEW
		tilesH = dim[1]/TILEH

		streamCompact = StreamCompact(context, [device], tilesW*tilesH)

		shapeTiles = (tilesW, tilesH)

		self.tilelistLoad = TileList(context, [device], shapeTiles)
		self.tilelistBfs = TileList(context, [device], shapeTiles)
		self.tilelistEdges = TileList(context, [device], shapeTiles)
		self.tilelistBorder = TileList(context, [device], shapeTiles)

		self.gWorksizeTiles16 = clutil.roundUp(shape, lWorksizeTiles16)
		self.gWorksizeWaves = (width, height/WAVE_LENGTH)
		self.gWorksizeSingleWave = (width, height/(WAVES_PER_WORKGROUP*WAVE_LENGTH))

		self.argsLoadTiles = [
			None, #dList
			self.dImg,
			self.dUp,
			self.dDown,
			self.dLeft,
			self.dRight,
			cl.LocalMemory(szInt*(TILEH+2)*(TILEW+2)),
			self.tilelistLoad.dTiles

		]
		self.argsInitBfs = [
			None, #dList
			self.dExcess,
			cl.LocalMemory(szInt*TILEH*TILEW),
			self.dHeight,
			cl.LocalMemory(szInt*(2+4)),
			self.dDown, self.dUp, self.dRight, self.dLeft,
			self.dCanDown, self.dCanUp, self.dCanRight, self.dCanLeft,
			cl.LocalMemory(szChar*WAVE_BREDTH*WAVES_PER_WORKGROUP),
			cl.LocalMemory(szChar*WAVE_BREDTH*WAVES_PER_WORKGROUP),
			cl.LocalMemory(szChar*WAVE_BREDTH*WAVES_PER_WORKGROUP),
			cl.LocalMemory(szChar*WAVE_BREDTH*WAVES_PER_WORKGROUP),
			self.tilelistBfs.dTiles,
			None
		]

		self.argsPushUpDown = [
			self.tilelistLoad.dList,
			self.dDown,
			self.dUp,
			self.dHeight,
			self.dExcess,
			self.dBorder,
			cl.LocalMemory(szInt*1),
			self.tilelistBorder.dTiles,
			None
		]

		self.argsPushLeftRight = [
			self.tilelistLoad.dList,
			self.dExcess,
			cl.LocalMemory(szFloat*TILEH*(TILEW+1)),
			dHeight,
			cl.LocalMemory(szFloat*TILEH*(TILEW+1)),
			self.dRight,
			self.dLeft,
			self.dBorder,
			cl.LocalMemory(szInt*1),
			self.tilelistBorder.dTiles,
			None
		]

		self.argsAddBorder = [
			self.tilelistBorder.dList,
			self.dBorder,
			self.dExcess,
			None,
		]

	def intratile_gaps(self):
		gWorksizeBfs = (WAVE_BREDTH*self.tilelistBfs.length, WAVE_LENGTH)

		tilelistEdges.increment()

		args = [
			tilelistBfs.dList,
			dHeight,
			cl.LocalMemory(szInt*(WAVE_BREDTH+2)*(WAVE_LENGTH*WAVES_PER_WORKGROUP+2)),
			dCanDown, dCanUp, dCanRight, dCanLeft,
			cl.LocalMemory(szInt*1),
			tilelistEdges.dTiles,
			np.int32(tilelistEdges.iteration)
		]

		kernBfs(queue, gWorksizeBfs, lWorksizeWaves, *args).wait()


	def intertile_gaps(self):
		lWorksizeBfs = (lWorksizeWaves[0], )
		gWorksizeBfs = (lWorksizeWaves[0]*int(tilelistEdges.length), )

		tilelistBfs.increment()

		args = [
			tilelistEdges.dList,
			cl.LocalMemory(szInt*3),
			dHeight,
			dCanDown, dCanUp, dCanRight, dCanLeft,
			np.int32(tilelistBfs.iteration),
			tilelistBfs.dTiles
		]

		kernBfsInterTile(queue, gWorksizeBfs, lWorksizeBfs, *args).wait()

	def startBfs(dList, length):
		tilelistBfs.increment()
		argsInitBfs[18] = np.int32(tilelistBfs.iteration)

		argsInitBfs[3] = dHeight

		argsInitBfs[0] =  dList

		gWorksize = (lWorksizeWaves[0]*int(length), lWorksizeWaves[1])
		kernInitBfs(queue, gWorksize, lWorksizeWaves, *argsInitBfs).wait()

		while True:
			tilelistBfs.flag()

			if tilelistBfs.length > 0:
				intratile_gaps()
			else:
				break;

			tilelistEdges.flag()

			if tilelistEdges.length > 0:
				intertile_gaps()
			else:
				break;

	def relabel():
		global dHeight, dHeight2

		kernRelabel(queue, gWorksizeTiles16, lWorksizeTiles16, dDown, dRight, dUp, dLeft, dExcess, dHeight, dHeight2)

		tmpHeight = dHeight
		dHeight = dHeight2
		dHeight2 = tmpHeight

	def push():
		#dHeight and dHeight2 swap after BFS
		argsPushUpDown[3] = dHeight
		argsPushLeftRight[3] = dHeight

		gWorksizeList = (lWorksizeWaves[0]*tilelistLoad.length, lWorksizeWaves[1])

		tilelistBorder.increment()
		argsPushUpDown[8] = np.int32(tilelistBorder.iteration)
		kernPushDown(queue, gWorksizeList, lWorksizeWaves, *argsPushUpDown).wait()
		tilelistBorder.flag()
		if tilelistBorder.length:
			argsAddBorder[3] = np.int32(0)
			gWorksizeBorder2 = (lWorksizeWaves[0]*tilelistBorder.length, )
			kernAddBorder(queue, gWorksizeBorder2, lWorksizeBorderAdd, *argsAddBorder).wait()


		tilelistBorder.increment()
		argsPushUpDown[8] = np.int32(tilelistBorder.iteration)
		kernPushUp(queue, gWorksizeList, lWorksizeWaves, *argsPushUpDown).wait()
		tilelistBorder.flag()
		if tilelistBorder.length:
			argsAddBorder[3] = np.int32(1)
			gWorksizeBorder2 = (lWorksizeWaves[0]*tilelistBorder.length, )
			kernAddBorder(queue, gWorksizeBorder2, lWorksizeBorderAdd, *argsAddBorder).wait()


		tilelistBorder.increment()
		argsPushLeftRight[10] = np.int32(tilelistBorder.iteration)
		kernPushRight(queue, gWorksizeList, lWorksizeWaves, *argsPushLeftRight).wait()
		tilelistBorder.flag()
		if tilelistBorder.length:
			argsAddBorder[3] = np.int32(2)
			gWorksizeBorder2 = (lWorksizeWaves[0]*tilelistBorder.length, )
			kernAddBorder(queue, gWorksizeBorder2, lWorksizeBorderAdd, *argsAddBorder).wait()


		tilelistBorder.increment()
		argsPushLeftRight[10] = np.int32(tilelistBorder.iteration)
		kernPushLeft(queue, gWorksizeList, lWorksizeWaves, *argsPushLeftRight).wait()
		tilelistBorder.flag()
		if tilelistBorder.length:
			argsAddBorder[3] = np.int32(3)
			gWorksizeBorder2 = (lWorksizeWaves[0]*tilelistBorder.length, )
			kernAddBorder(queue, gWorksizeBorder2, lWorksizeBorderAdd, *argsAddBorder).wait()


	def constructor():
		pass
#		write kernel
#		self.iteration = 1
#
#		hDown[:] = 0
#		hUp[:] = 0
#		hRight[:] = 0
#		hLeft[:] = 0
#		hHeight[:] = 0
#		hHeight2[:] = 0
#
#		cl.enqueue_copy(queue, dDown, hDown).wait()
#		cl.enqueue_copy(queue, dUp, hUp).wait()
#		cl.enqueue_copy(queue, dRight, hRight).wait()
#		cl.enqueue_copy(queue, dLeft, hLeft).wait()
#		cl.enqueue_copy(queue, dHeight, hHeight).wait()
#		cl.enqueue_copy(queue, dHeight2, hHeight2).wait()
#
#		tilelistLoad.reset()
#		tilelistBfs.reset()
#		tilelistEdges.reset()
#		tilelistBorder.reset()

	def isCompleted():
		cl.enqueue_copy(queue, dIsCompleted, np.int32(True))

		args = [
			tilelistLoad.dList,
			dExcess,
			dHeight,
			cl.LocalMemory(szInt*1),
			dIsCompleted
		]

		gWorksizeList = (lWorksizeWaves[0]*tilelistLoad.length, lWorksizeWaves[1])
		kernCheckCompletion(queue, gWorksizeList, lWorksizeWaves, *args)

		cl.enqueue_copy(queue, hIsCompleted, dIsCompleted).wait()

		return True if hIsCompleted[0] else False

	def cut(self, dExcess):
		#in the future only accept with dExcess and build tile list from there
		#hit the ground running with dExcess already calculated everywhere
		argsInitGC = [
			dExcess,
			cl.LocalMemory(szInt*1),
			tilelistLoad.dTiles,
			np.int32(tilelistLoad.iteration),
		]

		kernInitGC(queue, gWorksizeWaves, lWorksizeWaves, *argsInitGC).wait()

		tilelistLoad.flag()
		gWorksizeList = (lWorksizeWaves[0]*tilelistLoad.length, lWorksizeWaves[1])
		argsLoadTiles[0] = tilelistLoad.dList
		kernLoadTiles(queue, gWorksizeList, lWorksizeWaves, *argsLoadTiles)

		startBfs(tilelistLoad.dList, tilelistLoad.length)

		iteration = 1
		globalbfs = 5

		while True:
			print 'iteration:', iteration

			push()

			if iteration%globalbfs == 0:
				tilelistLoad.flagLogical(tilelistBorder.dTiles,
					StreamCompact.OPERATOR_EQUAL, -1,
					StreamCompact.OPERATOR_GT, tilelistBorder.iteration-globalbfs*4,
					StreamCompact.LOGICAL_AND
				)

				if tilelistLoad.length > 0:
					print '-- loading {0} tiles'.format(tilelistLoad.length)
					gWorksizeList = (lWorksizeWaves[0]*tilelistLoad.length, lWorksizeWaves[1])
					argsLoadTiles[0] = tilelistLoad.dList
					kernLoadTiles(queue, gWorksizeList, lWorksizeWaves, *argsLoadTiles).wait()

				tilelistLoad.flag(StreamCompact.OPERATOR_GT, -1)
				startBfs(tilelistLoad.dList, tilelistLoad.length)

				if isCompleted():
					window.updateCanvas()
					return

				print 'active tiles:, ', tilelistLoad.length
			else:
		#		pass
				relabel()

			iteration += 1

			window.updateCanvas()


if __name__ == "__main__":
	img = Image.open("/Users/marcdeklerk/msc/code/dataset/processed/source/800x600/GT04.png")
	if img.mode != 'RGBA':
		img = img.convert('RGBA')

	shapeNP = (img.size[1], img.size[0])
	hImg = padArray2D(np.array(img).view(np.uint32).squeeze(), roundUp(shapeNP, lWorksizeTiles16), 'edge')

	width = hImg.shape[1]
	height = hImg.shape[0]
	length = width*height
	size = (width, height)
	shape = (width, height)
	shapeT = (height, width)
	shapeNp = (height, width)
	shapeNpT = (width, height)

	app = QtGui.QApplication(sys.argv)
	canvas = CLCanvas(shape)
	window = CLWindow(canvas)

	window.resize(1000, 700)

	hSrc = np.load('scoreFg.npy').reshape(shapeNp)
	hSink = np.load('scoreBg.npy').reshape(shapeNp)

	hWeightMin = 0
	hWeightMax = LAMBDA * 1.0/EPSILON

	class TileListFilter():
		class Filter(Colorize.Filter):
			def __init__(self, kern, format, hues, tileflags):
				Colorize.Filter.__init__(self, kern, format, (0, tileflags.iteration), hues)

				self.tileflags = tileflags

			def execute(self, queue, args):
				range = np.array([self.tileflags.iteration-1, self.tileflags.iteration], np.int32)

				self.kern(queue, gWorksizeTiles16, lWorksizeTiles16, range, self.hues, *args)

		def __init__(self, canvas):
			filename = os.path.join(os.path.dirname(__file__), 'graphcut.cl')
			program = createProgram(canvas.context, canvas.devices, options, filename)

			self.kernel = cl.Kernel(program, 'tilelist')

		def factory(self, tileflags, hues=None):
			if hues == None:
				hues = Colorize.HUES.STANDARD

			return TileListFilter.Filter(self.kernel, (Buffer2D, np.int32), hues, tileflags)

	class TransposeColorize():
		class Filter(Colorize.Filter):
			def __init__(self, kern, format, range, hues):
				Colorize.Filter.__init__(self, kern, format, range, hues)

			def execute(self, queue, args):
				args.append(args[-1].dim)

				self.kern(queue, gWorksizeWaves, lWorksizeWaves, self.range, self.hues, *args)

		def __init__(self, canvas):
			filename = os.path.join(os.path.dirname(__file__), 'graphcut.cl')
			program = createProgram(canvas.context, canvas.devices, options, filename)

			self.kernel = cl.Kernel(program, 'tranpose')

		def factory(self, range, hues=None):
			if hues == None:
				hues = Colorize.HUES.STANDARD

			return TransposeColorize.Filter(self.kernel, (Buffer2D, np.float32), range, hues)

	tilelistfilter = TileListFilter(canvas)
	transposefilter = TransposeColorize(canvas)
	colorize = Colorize(canvas)

	filter = colorize.factory((Buffer2D, np.float32), (0, 50), hues=Colorize.HUES.REVERSED)
	window.addLayer('excess', dExcess, filter=filter)

	filter = colorize.factory((Buffer2D, np.int32), (1, 144), hues=Colorize.HUES.REVERSED)
	window.addLayer('height', dHeight, filter=filter)
	window.addLayer('height2/bfs', dHeight2, filter=filter)

	filter = tilelistfilter.factory(tilelistBfs, hues=Colorize.HUES.REVERSED)
	window.addLayer('tiles Bfs', tilelistBfs.dTiles, filter=filter)

	filter = tilelistfilter.factory(tilelistLoad, hues=Colorize.HUES.REVERSED)
	window.addLayer('tiles Load', tilelistLoad.dTiles, filter=filter)

	filter = tilelistfilter.factory(tilelistBorder, hues=Colorize.HUES.REVERSED)
	window.addLayer('tiles Border', tilelistBorder.dTiles, filter=filter)

	filter = tilelistfilter.factory(tilelistEdges, hues=Colorize.HUES.REVERSED)
	window.addLayer('tiles Edges', tilelistEdges.dTiles, filter=filter)

	filter = colorize.factory((Buffer2D, np.float32), (0.001, hWeightMax), hues=Colorize.HUES.REVERSED)
	window.addLayer('up', dUp, filter=filter)
	window.addLayer('down', dDown, filter=filter)

	filter = transposefilter.factory((0.001, hWeightMax), hues=Colorize.HUES.REVERSED)
	window.addLayer('left', dLeft, filter=filter)
	window.addLayer('right', dRight, filter=filter)

	window.addLayer('img', dImg)








timer = QtCore.QTimer()
timer.timeout.connect(next)

window.addButton("push", push)
window.addButton("relabel", relabel)
window.addButton("next", next)
window.addButton("start", functools.partial(timer.start, 0))
window.addButton("stop", timer.stop)
window.addButton("intercept", intercept)
window.addButton("init", init)
window.addButton("reset", reset)

window.show()
sys.exit(app.exec_())


