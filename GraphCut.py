__author__ = 'marcdeklerk'

import os
import numpy as np
import Image
import pyopencl as cl
import msclib, msclib.clutil as clutil
from msclib import context, queue, device
#from msclib import Colorize
from StreamCompact import StreamCompact, TileList
from clutil import createProgram, cl, roundUp, ceil_divi, padArray2D, Buffer2D
import functools

from PyQt4 import QtCore, QtGui, QtOpenGL
from CLWindow import CLWindow
from CLCanvas import CLCanvas
from CLCanvas import Filter as CLFilter
from Colorize import Colorize
import sys
import time

szFloat = np.dtype(np.float32).itemsize
szInt = np.dtype(np.int32).itemsize
szChar = np.dtype(np.uint8).itemsize

cm = cl.mem_flags

img = Image.open("/Users/marcdeklerk/msc/code/dataset/processed/source/800x600/GT04.png")
if img.mode != 'RGBA':
	img = img.convert('RGBA')

lWorksizeTiles16 = (16, 16)

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

LAMBDA = 60
EPSILON = 0.05
MAX_HEIGHT = height*width
#MAX_HEIGHT = 240

wave_bredth = 32
wave_length = 8
waves_per_workgroup = 4

tilesW = width/wave_bredth
tilesH = height/(waves_per_workgroup*wave_length)

options = []
options += ['-D MAX_HEIGHT='+repr(MAX_HEIGHT)]
options += ['-D LAMBDA='+repr(LAMBDA)]
options += ['-D EPSILON='+repr(EPSILON)]
options += ['-D TILESW='+str(tilesW)]
options += ['-D TILESH='+str(tilesH)]

context = canvas.context
devices = context.get_info(cl.context_info.DEVICES)
queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

dir = os.path.dirname(__file__)
def build():
	global kernInitGC, kernPushDown, kernPushUp, kernRelabel, kernAddBorder
	global kernViewBorder, kernViewActive , kernBfs, kernBfsInterTile , kernInitBfs , kernClear, kernLoadTiles
	global kernInitPush, kernPushRight, kernPushLeft, kernCheckCompletion

	program = clutil.createProgram(context, devices, options, os.path.join(dir, 'graphcut.cl'))
	kernInitGC = cl.Kernel(program, 'init_gc')
	kernPushDown = cl.Kernel(program, 'push_down')
	kernPushUp = cl.Kernel(program, 'push_up')
	kernRelabel = cl.Kernel(program, 'relabel')
	kernAddBorder = cl.Kernel(program, 'add_border')
	kernBfs = cl.Kernel(program, 'bfs_intratile')
	kernBfsInterTile = cl.Kernel(program, 'bfs_intertile')
	kernInitBfs = cl.Kernel(program, 'init_bfs')
	kernLoadTiles = cl.Kernel(program, 'load_tiles')
	kernPushRight = cl.Kernel(program, 'push_right')
	kernPushLeft = cl.Kernel(program, 'push_left')
	kernCheckCompletion = cl.Kernel(program, 'check_completion')

build()



hSrc = np.load('scoreFg.npy').reshape(shapeNp)
hSink = np.load('scoreBg.npy').reshape(shapeNp)

hUp = np.empty(shapeNp, np.float32)
hDown = np.empty(shapeNp, np.float32)
hLeft = np.empty(shapeNpT, np.float32)
hRight = np.empty(shapeNpT, np.float32)
hHeight = np.zeros(shapeNp, np.int32)
hExcess = hSink - hSrc
hBorder = np.empty((height/waves_per_workgroup, width), np.float32)
hHeight2 = np.zeros(shapeNp, np.int32)
hCanDown = np.empty((hDown.shape[0]/32, hDown.shape[1]) , np.uint32)
hCanUp = np.empty((hUp.shape[0]/32, hUp.shape[1]) , np.uint32)
hCanRight = np.empty((hRight.shape[0]/32, hRight.shape[1]) , np.uint32)
hCanLeft = np.empty((hLeft.shape[0]/32, hLeft.shape[1]) , np.uint32)
hIsCompleted = np.empty((1,), np.int32)

dImg = Buffer2D(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hImg)
dUp = Buffer2D(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hUp)
dDown = Buffer2D(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hDown)
dLeft = Buffer2D(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hLeft)
dRight = Buffer2D(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hRight)
dHeight = Buffer2D(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hHeight)
dHeight2 = Buffer2D(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hHeight2)
dExcess = Buffer2D(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hExcess)
dBorder = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hBorder)
dCanUp = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hUp)
dCanDown = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hCanDown)
dCanLeft = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hCanLeft)
dCanRight = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hCanRight)
dIsCompleted = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hIsCompleted)

streamCompact = StreamCompact(context, [device], tilesW*tilesH)

shapeTiles = (tilesW, tilesH)

tilelistLoad = TileList(context, [device], shapeTiles)
tilelistBfs = TileList(context, [device], shapeTiles)
tilelistEdges = TileList(context, [device], shapeTiles)
tilelistBorder = TileList(context, [device], shapeTiles)

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

lWorksizeTiles16 = (16, 16)
gWorksizeTiles16 = clutil.roundUp(shape, lWorksizeTiles16)

lWorksizeWaves = (wave_bredth, waves_per_workgroup)
gWorksizeWaves = (width, height/wave_length)

lWorksizeSingleWave = (wave_bredth, 1)
gWorksizeSingleWave = (width, height/(waves_per_workgroup*wave_length))

lWorksizeBorderAdd = (wave_bredth, )

argsLoadTiles = [
	None, #dList
	dImg,
	dUp,
	dDown,
	dLeft,
	dRight,
	cl.LocalMemory(szInt*(waves_per_workgroup*wave_length+2)*(wave_bredth+2)),
	tilelistLoad.dTiles
]

def constructor():
	global iteration

	iteration = 1

	hDown[:] = 0
	hUp[:] = 0
	hRight[:] = 0
	hLeft[:] = 0
	hHeight[:] = 0
	hHeight2[:] = 0

	cl.enqueue_copy(queue, dDown, hDown).wait()
	cl.enqueue_copy(queue, dUp, hUp).wait()
	cl.enqueue_copy(queue, dRight, hRight).wait()
	cl.enqueue_copy(queue, dLeft, hLeft).wait()
	cl.enqueue_copy(queue, dHeight, hHeight).wait()
	cl.enqueue_copy(queue, dHeight2, hHeight2).wait()

	tilelistLoad.reset()
	tilelistBfs.reset()
	tilelistEdges.reset()
	tilelistBorder.reset()

constructor()

def init():
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

	window.updateCanvas()


argsInitBfs = [
	None, #dList
	dExcess,
	cl.LocalMemory(szInt*(wave_bredth)*(wave_length*waves_per_workgroup)),
	dHeight,
	cl.LocalMemory(szInt*(2+4)),
	dDown, dUp, dRight, dLeft,
	dCanDown, dCanUp, dCanRight, dCanLeft,
	cl.LocalMemory(szChar*lWorksizeWaves[0]*lWorksizeWaves[1]),
	cl.LocalMemory(szChar*lWorksizeWaves[0]*lWorksizeWaves[1]),
	cl.LocalMemory(szChar*lWorksizeWaves[0]*lWorksizeWaves[1]),
	cl.LocalMemory(szChar*lWorksizeWaves[0]*lWorksizeWaves[1]),
	tilelistBfs.dTiles,
	None
	]


def intratile_gaps():
	gWorksizeBfs = (lWorksizeWaves[0]*int(tilelistBfs.length), lWorksizeWaves[1])

	tilelistEdges.increment()

	args = [
		tilelistBfs.dList,
		dHeight,
		cl.LocalMemory(szInt*(wave_bredth+2)*(wave_length*waves_per_workgroup+2)),
		dCanDown, dCanUp, dCanRight, dCanLeft,
		cl.LocalMemory(szInt*1),
		tilelistEdges.dTiles,
		np.int32(tilelistEdges.iteration)
	]

	kernBfs(queue, gWorksizeBfs, lWorksizeWaves, *args).wait()


def intertile_gaps():
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

def intercept():
	pass

def relabel():
	global dHeight, dHeight2

	kernRelabel(queue, gWorksizeTiles16, lWorksizeTiles16, dDown, dRight, dUp, dLeft, dExcess, dHeight, dHeight2)

	tmpHeight = dHeight
	dHeight = dHeight2
	dHeight2 = tmpHeight

argsPushUpDown = [
	tilelistLoad.dList,
	dDown,
	dUp,
	dHeight,
	dExcess,
	dBorder,
	cl.LocalMemory(szInt*1),
	tilelistBorder.dTiles,
	None
]

argsPushLeftRight = [
	tilelistLoad.dList,
	dExcess,
	cl.LocalMemory(4*waves_per_workgroup*wave_length*(wave_bredth+1)),
	dHeight,
	cl.LocalMemory(4*waves_per_workgroup*wave_length*(wave_bredth+1)),
	dRight,
	dLeft,
	dBorder,
	cl.LocalMemory(szInt*1),
	tilelistBorder.dTiles,
	None
]

argsAddBorder = [
	tilelistBorder.dList,
	dBorder,
	dExcess,
	None,
]


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

iteration = 1
globalbfs = 5

def next():
	print 'iteration:', iteration
	global iteration

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
			timer.stop()
			window.updateCanvas()

			print 'complete'

		print 'active tiles:, ', tilelistLoad.length
	else:
#		pass
		relabel()

	iteration += 1

	window.updateCanvas()

timer = QtCore.QTimer()
timer.timeout.connect(next)

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

def reset():
	build()
	constructor()
	init()

window.addButton("push", push)
window.addButton("relabel", relabel)
window.addButton("next", next)
window.addButton("start", functools.partial(timer.start, 0))
window.addButton("stop", timer.stop)
window.addButton("intercept", intercept)
window.addButton("init", init)
window.addButton("reset", reset)

init()

window.show()
sys.exit(app.exec_())


