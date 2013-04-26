__author__ = 'marcdeklerk'

import sys, os
import numpy as np
import Image
import pyopencl as cl
from StreamCompact import StreamCompact, IncrementalTileList
from clutil import createProgram, roundUp, padArray2D, Buffer2D

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

class GraphCut:
	lWorksizeTiles16 = (16, 16)
	lWorksizeSingleWave = (WAVE_BREDTH, 1)
	lWorksizeWaves = (WAVE_BREDTH, WAVES_PER_WORKGROUP)
	lWorksizeBorderAdd = (WAVE_BREDTH, )

	LAMBDA_DEFAULT = 60
	EPSILON = 0.05
	BFS_DEFAULT = 5

	def __init__(self, context, devices, dImg, lamda=LAMBDA_DEFAULT):
		MAX_HEIGHT = height*width

		self.lamda = lamda

		dim = dImg.dim
		tilesW = dim[0]/TILEW
		tilesH = dim[1]/TILEH

		options = []
		options += ['-D MAX_HEIGHT='+repr(MAX_HEIGHT)]
		options += ['-D LAMBDA='+repr(lamda)]
		options += ['-D EPSILON='+repr(GraphCut.EPSILON)]
		options += ['-D TILESW='+str(tilesW)]
		options += ['-D TILESH='+str(tilesH)]

		self.queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

		dir = os.path.dirname(__file__)

		program = createProgram(context, devices, options, os.path.join(dir, 'graphcut.cl'))
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

		size = sz32*dim[0]*dim[1]
		sizeSingleWave = sz32*dim[0]*(dim[1]/TILEH)
		sizeCompressedWave = sz32*dim[0]*(dim[1]/TILEH)

		self.hIsCompleted = np.array((1, ), np.int32)

		self.dImg = dImg
		self.dUp = Buffer2D(context, cm.READ_WRITE, dim=dim, dtype=np.float32)
		self.dDown = Buffer2D(context, cm.READ_WRITE, dim=dim, dtype=np.float32)
		self.dLeft = Buffer2D(context, cm.READ_WRITE, dim=dim, dtype=np.float32)
		self.dRight = Buffer2D(context, cm.READ_WRITE, dim=dim, dtype=np.float32)
		self.dHeight = Buffer2D(context, cm.READ_WRITE, dim=dim, dtype=np.int32)
		self.dHeight2 = Buffer2D(context, cm.READ_WRITE, dim=dim, dtype=np.int32)
		self.dBorder = cl.Buffer(context, cm.READ_WRITE, sizeSingleWave)
		self.dCanUp = cl.Buffer(context, cm.READ_WRITE, sizeSingleWave)
		self.dCanDown = cl.Buffer(context, cm.READ_WRITE, sizeSingleWave)
		self.dCanLeft = cl.Buffer(context, cm.READ_WRITE, sizeSingleWave)
		self.dCanRight = cl.Buffer(context, cm.READ_WRITE, sizeSingleWave)
		self.dIsCompleted = cl.Buffer(context, cm.READ_WRITE, szInt)

		streamCompact = StreamCompact(context, devices, tilesW*tilesH)

		shapeTiles = (tilesW, tilesH)

		self.tilelistLoad = IncrementalTileList(context, devices, shapeTiles)
		self.tilelistBfs = IncrementalTileList(context, devices, shapeTiles)
		self.tilelistEdges = IncrementalTileList(context, devices, shapeTiles)
		self.tilelistBorder = IncrementalTileList(context, devices, shapeTiles)

		self.gWorksizeTiles16 = roundUp(shape, GraphCut.lWorksizeTiles16)
		self.gWorksizeWaves = (width, height/WAVE_LENGTH)
		self.gWorksizeSingleWave = (width, height/(WAVES_PER_WORKGROUP*WAVE_LENGTH))

	def intratile_gaps(self):
		gWorksizeBfs = (WAVE_BREDTH*self.tilelistBfs.length, WAVE_LENGTH)

		args = [
			self.tilelistBfs.dList,
			self.dHeight,
			cl.LocalMemory(szInt*(WAVE_BREDTH+2)*(WAVE_LENGTH*WAVES_PER_WORKGROUP+2)),
			self.dCanDown, self.dCanUp, self.dCanRight, self.dCanLeft,
			cl.LocalMemory(szInt*1),
			self.tilelistEdges.dTiles,
			np.int32(self.tilelistEdges.increment())
		]
		self.kernBfsIntraTile(self.queue, gWorksizeBfs, self.lWorksizeWaves, *args).wait()

	def intertile_gaps(self):
		lWorksizeBfs = (WAVE_BREDTH, )
		gWorksizeBfs = (WAVE_BREDTH*self.tilelistEdges.length, )

		args = [
			self.tilelistEdges.dList,
			cl.LocalMemory(szInt*3),
			self.dHeight,
			self.dCanDown, self.dCanUp, self.dCanRight, self.dCanLeft,
			np.int32(self.tilelistBfs.increment()),
			self.tilelistBfs.dTiles
		]
		self.kernBfsInterTile(self.queue, gWorksizeBfs, lWorksizeBfs, *args).wait()

	def startBfs(self, dExcess):
		gWorksize = (WAVE_BREDTH*int(self.tilelistLoad.length), WAVES_PER_WORKGROUP)

		args = [
			self.tilelistLoad.dList,
			dExcess,
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
			np.int32(self.tilelistBfs.increment())
		]
		self.kernInitBfs(self.queue, gWorksize, self.lWorksizeWaves, *args).wait()

		while True:
			self.tilelistBfs.build()

			if self.tilelistBfs.length > 0:
				self.intratile_gaps()
			else:
				break;

			self.tilelistEdges.build()

			if self.tilelistEdges.length > 0:
				self.intertile_gaps()
			else:
				break;

	def relabel(self, dExcess):
		args = [
			self.dDown,
			self.dRight,
			self.dUp,
			self.dLeft,
			dExcess,
			self.dHeight,
			self.dHeight2
		]

		self.kernRelabel(self.queue, self.gWorksizeTiles16, GraphCut.lWorksizeTiles16, *args)

		tmpHeight = self.dHeight
		self.dHeight = self.dHeight2
		self.dHeight2 = tmpHeight

	def push(self, dExcess):
		gWorksize = (WAVE_BREDTH*self.tilelistLoad.length, WAVES_PER_WORKGROUP)

		self.argsPushUpDown = [
			self.tilelistLoad.dList,
			self.dDown, self.dUp,
			self.dHeight,
			dExcess,
			self.dBorder,
			cl.LocalMemory(szInt*1),
			self.tilelistBorder.dTiles,
			None
		]

		self.argsPushLeftRight = [
			self.tilelistLoad.dList,
			dExcess,
			cl.LocalMemory(szFloat*TILEH*(TILEW+1)),
			self.dHeight,
			cl.LocalMemory(szFloat*TILEH*(TILEW+1)),
			self.dRight, self.dLeft,
			self.dBorder,
			cl.LocalMemory(szInt*1),
			self.tilelistBorder.dTiles,
			None
		]

		self.argsAddBorder = [
			self.tilelistBorder.dList,
			self.dBorder,
			dExcess,
			None,
		]

		self.argsPushUpDown[8] = np.int32(self.tilelistBorder.increment())
		self.kernPushDown(self.queue, gWorksize, self.lWorksizeWaves, *self.argsPushUpDown).wait()

		self.tilelistBorder.build()
		if self.tilelistBorder.length:
			self.argsAddBorder[3] = np.int32(0)
			gWorksizeBorder = (WAVE_BREDTH*self.tilelistBorder.length, )
			self.kernAddBorder(self.queue, gWorksizeBorder, self.lWorksizeBorderAdd, *self.argsAddBorder).wait()

		self.argsPushUpDown[8] = np.int32(self.tilelistBorder.increment())
		self.kernPushUp(self.queue, gWorksize, self.lWorksizeWaves, *self.argsPushUpDown).wait()

		self.tilelistBorder.build()
		if self.tilelistBorder.length:
			self.argsAddBorder[3] = np.int32(1)
			gWorksizeBorder = (WAVE_BREDTH*self.tilelistBorder.length, )
			self.kernAddBorder(self.queue, gWorksizeBorder, self.lWorksizeBorderAdd, *self.argsAddBorder).wait()

		self.argsPushLeftRight[10] = np.int32(self.tilelistBorder.increment())
		self.kernPushRight(self.queue, gWorksize, self.lWorksizeWaves, *self.argsPushLeftRight).wait()

		self.tilelistBorder.build()
		if self.tilelistBorder.length:
			self.argsAddBorder[3] = np.int32(2)
			gWorksizeBorder = (WAVE_BREDTH*self.tilelistBorder.length, )
			self.kernAddBorder(self.queue, gWorksizeBorder, self.lWorksizeBorderAdd, *self.argsAddBorder).wait()

		self.argsPushLeftRight[10] = np.int32(self.tilelistBorder.increment())
		self.kernPushLeft(self.queue, gWorksize, self.lWorksizeWaves, *self.argsPushLeftRight).wait()

		self.tilelistBorder.build()
		if self.tilelistBorder.length:
			self.argsAddBorder[3] = np.int32(3)
			gWorksizeBorder = (WAVE_BREDTH*self.tilelistBorder.length, )
			self.kernAddBorder(self.queue, gWorksizeBorder, self.lWorksizeBorderAdd, *self.argsAddBorder).wait()


	def reset(self):
#		pass
		self.iteration = 1

#		hDown[:] = 0
#		hUp[:] = 0
#		hRight[:] = 0
#		hLeft[:] = 0
#		hHeight[:] = 0
#		hHeight2[:] = 0

#		cl.enqueue_copy(queue, dDown, hDown).wait()
#		cl.enqueue_copy(queue, dUp, hUp).wait()
#		cl.enqueue_copy(queue, dRight, hRight).wait()
#		cl.enqueue_copy(queue, dLeft, hLeft).wait()
#		cl.enqueue_copy(self.queue, self.dHeight, self.hHeight).wait()
#		cl.enqueue_copy(queue, dHeight2, hHeight2).wait()

#		self.tilelistLoad.reset()
#		self.tilelistBfs.reset()
#		self.tilelistEdges.reset()
#		self.tilelistBorder.reset()

	def isCompleted(self, dExcess):
		cl.enqueue_copy(self.queue, self.dIsCompleted, np.int32(True))

		gWorksize = (WAVE_BREDTH*self.tilelistLoad.length, WAVES_PER_WORKGROUP)

		args = [
			self.tilelistLoad.dList,
			dExcess,
			self.dHeight,
			cl.LocalMemory(szInt*1),
			self.dIsCompleted
		]
		self.kernCheckCompletion(self.queue, gWorksize, self.lWorksizeWaves, *args)

		cl.enqueue_copy(self.queue, self.hIsCompleted, self.dIsCompleted)
		self.queue.finish()

		return True if self.hIsCompleted[0] else False

	def cut(self, dExcess, bfs=BFS_DEFAULT):
		loadIteration = self.tilelistLoad.iteration

		argsInitGC = [
			self.dUp,
			self.dDown,
			self.dLeft,
			self.dRight,
			dExcess,
			cl.LocalMemory(szInt*1),
			self.tilelistLoad.dTiles,
			np.int32(self.tilelistLoad.increment()),
			]

		argsLoadTiles = [
			self.tilelistLoad.dList,
			self.dImg,
			self.dUp, self.dDown, self.dLeft, self.dRight,
			cl.LocalMemory(szInt*(TILEH+2)*(TILEW+2))
		]

		self.kernInitGC(self.queue, self.gWorksizeWaves, self.lWorksizeWaves, *argsInitGC).wait()

		self.tilelistLoad.build()
		gWorksize = (WAVE_BREDTH*self.tilelistLoad.length, WAVES_PER_WORKGROUP)
		self.kernLoadTiles(self.queue, gWorksize, self.lWorksizeWaves, *argsLoadTiles).wait()

		self.startBfs(dExcess)

		iteration = 1

		while True:
			print 'iteration:', iteration

			self.push(dExcess)

			if iteration%bfs == 0:
				self.tilelistLoad.increment()

				self.tilelistLoad.incorporate(self.tilelistBorder.dTiles,
					StreamCompact.OPERATOR_LTE, loadIteration,
					StreamCompact.OPERATOR_GT, self.tilelistBorder.iteration-bfs*4,
					StreamCompact.LOGICAL_AND
				)

				self.tilelistLoad.build()
				if self.tilelistLoad.length > 0:
					print '-- loading {0} tiles'.format(self.tilelistLoad.length)
					gWorksize = (WAVE_BREDTH*self.tilelistLoad.length, WAVES_PER_WORKGROUP)
					self.kernLoadTiles(self.queue, gWorksize, self.lWorksizeWaves, *argsLoadTiles).wait()

				self.tilelistLoad.build(StreamCompact.OPERATOR_GT, loadIteration)
				self.startBfs(dExcess)

				if self.isCompleted(dExcess):
					window.updateCanvas()
					return

				print 'active tiles:, ', self.tilelistLoad.length
			else:
				self.relabel(dExcess)

			iteration += 1

	def reset(self):
		pass


if __name__ == "__main__":
	from PyQt4 import QtCore, QtGui
	from CLWindow import CLWindow
	from CLCanvas import CLCanvas
	from Colorize import Colorize
	import functools

	img = Image.open("/Users/marcdeklerk/msc/code/dataset/processed/source/800x600/GT04.png")
	if img.mode != 'RGBA':
		img = img.convert('RGBA')

	shapeNP = (img.size[1], img.size[0])
	hImg = padArray2D(np.array(img).view(np.uint32).squeeze(), roundUp(shapeNP, GraphCut.lWorksizeTiles16), 'edge')

	width = hImg.shape[1]
	height = hImg.shape[0]
	size = (width, height)
	shape = (width, height)
	shapeT = (height, width)
	shapeNp = (height, width)
	shapeNpT = (width, height)

	app = QtGui.QApplication(sys.argv)
	canvas = CLCanvas(shape)
	window = CLWindow(canvas)

	context = canvas.context

	dImg = Buffer2D(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hImg)

	devices = context.get_info(cl.context_info.DEVICES)

	gc = GraphCut(context, devices, dImg)

	window.resize(1000, 700)

	hSrc = np.load('scoreFg.npy').reshape(shapeNp)
	hSink = np.load('scoreBg.npy').reshape(shapeNp)

	dExcess = Buffer2D(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=(hSink-hSrc))

	hWeightMin = 0
	hWeightMax = gc.lamda * 1.0/GraphCut.EPSILON

	options = [
		'-D IMAGEW={0}'.format(shape[0]),
		'-D IMAGEH={0}'.format(shape[1]),
		'-D TILESW='+str(gc.tilelistLoad.shape[0]),
		'-D TILESH='+str(gc.tilelistLoad.shape[1])
	]
	filename = os.path.join(os.path.dirname(__file__), 'graphcut_filter.cl')
	program = createProgram(canvas.context, canvas.devices, options, filename)

	kernTranspose = cl.Kernel(program, 'tranpose')
	kernTileList = cl.Kernel(program, 'tilelist')

	class TileListFilter():
		class Filter(Colorize.Filter):
			def __init__(self, kern, format, hues, tileflags):
				Colorize.Filter.__init__(self, kern, format, (0, tileflags.iteration), hues)

				self.tileflags = tileflags

			def execute(self, queue, args):
				range = np.array([self.tileflags.iteration-1, self.tileflags.iteration], np.int32)

				kernTileList(queue, gc.gWorksizeTiles16, GraphCut.lWorksizeTiles16, range, self.hues, *args)

		def __init__(self, canvas):
			pass

		def factory(self, tileflags, hues=None):
			if hues == None:
				hues = Colorize.HUES.STANDARD

			return TileListFilter.Filter(kernTileList, (Buffer2D, np.int32), hues, tileflags)

	class TransposeColorize():
		class Filter(Colorize.Filter):
			def __init__(self, kern, format, range, hues):
				Colorize.Filter.__init__(self, kern, format, range, hues)

			def execute(self, queue, args):
				args.append(np.array(args[-1].dim, np.int32))

				kernTranspose(queue, gc.gWorksizeWaves, GraphCut.lWorksizeWaves, self.range, self.hues, *args)

		def __init__(self, canvas):
			pass

		def factory(self, range, hues=None):
			if hues == None:
				hues = Colorize.HUES.STANDARD

			return TransposeColorize.Filter(kernTranspose, (Buffer2D, np.float32), range, hues)

	tilelistfilter = TileListFilter(canvas)
	transposefilter = TransposeColorize(canvas)
	colorize = Colorize(canvas)

	filter = colorize.factory((Buffer2D, np.float32), (0, 50), hues=Colorize.HUES.REVERSED)
	window.addLayer('excess', dExcess, filter=filter)

	filter = colorize.factory((Buffer2D, np.int32), (1, 144), hues=Colorize.HUES.REVERSED)
	window.addLayer('height', gc.dHeight, filter=filter)
	window.addLayer('height2/bfs', gc.dHeight2, filter=filter)

	filter = tilelistfilter.factory(gc.tilelistBfs, hues=Colorize.HUES.REVERSED)
	window.addLayer('tiles Bfs', gc.tilelistBfs.dTiles, filter=filter)

	filter = tilelistfilter.factory(gc.tilelistLoad, hues=Colorize.HUES.REVERSED)
	window.addLayer('tiles Load', gc.tilelistLoad.dTiles, filter=filter)

	filter = tilelistfilter.factory(gc.tilelistBorder, hues=Colorize.HUES.REVERSED)
	window.addLayer('tiles Border', gc.tilelistBorder.dTiles, filter=filter)

	filter = tilelistfilter.factory(gc.tilelistEdges, hues=Colorize.HUES.REVERSED)
	window.addLayer('tiles Edges', gc.tilelistEdges.dTiles, filter=filter)

	window.addLayer('img', dImg)

	filter = colorize.factory((Buffer2D, np.float32), (0.001, hWeightMax), hues=Colorize.HUES.REVERSED)
	window.addLayer('up', gc.dUp, filter=filter)
	window.addLayer('down', gc.dDown, filter=filter)

	filter = transposefilter.factory((0.001, hWeightMax), hues=Colorize.HUES.REVERSED)
	window.addLayer('left', gc.dLeft, filter=filter)
	window.addLayer('right', gc.dRight, filter=filter)
#
	timer = QtCore.QTimer()
#	timer.timeout.connect(next)

	def reset():
		cl.enqueue_copy(gc.queue, dExcess, (hSink-hSrc))

	window.addButton("push", gc.push)
	window.addButton("relabel", gc.relabel)
	window.addButton("cut", functools.partial(gc.cut, dExcess, 5))
#	window.addButton("start", functools.partial(timer.start, 0))
#	window.addButton("stop", timer.stop)
	window.addButton("reset", reset)

	window.show()

	sys.exit(app.exec_())


