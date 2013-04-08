__author__ = 'marcdeklerk'

import os
import numpy as np
import Image
import pyopencl as cl
import msclib, msclib.clutil as clutil
from msclib import context, queue, device
#from msclib import Colorize
from StreamCompact import StreamCompact
from clutil import createProgram, cl, roundUp, ceil_divi, padArray2D, compareFormat
import functools

from PyQt4 import QtCore, QtGui, QtOpenGL
from CLWindow import CLWindow
from CLCanvas import CLCanvas, ChainedFilter
from Colorize import Colorize
import sys
import time

szFloat = np.dtype(np.float32).itemsize
szInt = np.dtype(np.int32).itemsize
szChar = np.dtype(np.uint8).itemsize

cm = cl.mem_flags

class List():
	def __init__(self, streamCompact):
		self.hLength = np.empty((1,), np.int32)

		self.dLength = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=self.hLength)
		self.dList = streamCompact.listFactory()

		self.isDirty = True

	@property
	def length(self):
		if self.isDirty:
			cl.enqueue_copy(queue, self.hLength, self.dLength).wait()
			self.isDirty = False

		return int(self.hLength[0])

	def build(self, queue, tileFlag):
		streamCompact.compact(queue, tileFlag.dFlags, self.dList, self.dLength)

		self.isDirty = True

class TileFlags():
	def __init__(self, streamCompact):
		self.iteration = 2 #0 and 1 reserved for True and False Flags

		self.dTiles = streamCompact.flagFactory()
		self.dFlags = streamCompact.flagFactory()

		hTiles = np.empty((streamCompact.capacity, ), np.int32)
		hTiles[:] = -1
		cl.enqueue_copy(queue, self.dTiles, hTiles).wait()

	def increment(self):
		self.iteration += 1

	def flag(self, queue, operator=None, operand=None):
		if operator == None: operator = StreamCompact.OPERATOR_EQUAL
		if operand == None:  operand = self.iteration

		streamCompact.flag(queue, self.dTiles, self.dFlags, operator, operand)

	def flagLogical(self, queue, dTiles2, operator1, operator2, operand1, operand2, logical):
		streamCompact.flagLogical(queue, self.dTiles, dTiles2, self.dFlags, operator1, operator2, operand1, operand2, logical)


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

options = []
options += ['-D MAX_HEIGHT='+repr(MAX_HEIGHT)]
options += ['-D LAMBDA='+repr(LAMBDA)]
options += ['-D EPSILON='+repr(EPSILON)]

context = canvas.context
devices = context.get_info(cl.context_info.DEVICES)
queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

#colorize = Colorize.Colorize(context, context.devices)

dir = os.path.dirname(__file__)
program = clutil.createProgram(context, devices, options, os.path.join(dir, 'graphcut.cl'))
kernInitNeighbourhood = cl.Kernel(program, 'initNeighbourhood')
kernInitGC = cl.Kernel(program, 'init_gc')
kernPushDown = cl.Kernel(program, 'pushDown')
#kernPushRight = cl.Kernel(program, 'pushRight')
kernPushUp = cl.Kernel(program, 'pushUp')
#kernPushLeft = cl.Kernel(program, 'pushLeft')
kernRelabel = cl.Kernel(program, 'relabel')
kernAddBorder = cl.Kernel(program, 'addBorder')
kernViewBorder = cl.Kernel(program, 'viewBorder')
kernViewActive = cl.Kernel(program, 'viewActive')
kernBfs = cl.Kernel(program, 'bfs_intratile')
kernBfsInterTile = cl.Kernel(program, 'bfs_intertile')
kernInitBfs = cl.Kernel(program, 'init_bfs')
kernBfsCompact = cl.Kernel(program, 'bfs_compact')
#kernMapActiveTiles = cl.Kernel(program, 'mapActiveTiles')
kernMapTileList = cl.Kernel(program, 'mapTileList')
kernClear = cl.Kernel(program, 'clear')
kernLoadTiles = cl.Kernel(program, 'load_tiles')
kernInitPush = cl.Kernel(program, 'init_push')

kernPushRight = cl.Kernel(program, 'test')
kernPushLeft = cl.Kernel(program, 'testL')

wave_bredth = 32
wave_length = 8
waves_per_workgroup = 4

tilesW = width/wave_bredth
tilesH = height/(waves_per_workgroup*wave_length)

hSrc = np.load('scoreFg.npy').reshape(shapeNp)
hSink = np.load('scoreBg.npy').reshape(shapeNp)

hUp = np.empty(shapeNp, np.float32)
hDown = np.empty(shapeNp, np.float32)
hLeft = np.empty(shapeNpT, np.float32)
hTranspose = np.empty(shapeNpT, np.float32)
hRight = np.empty(shapeNpT, np.float32)
hHeight = np.zeros(shapeNp, np.int32)
hExcess = hSink - hSrc
hBorder = np.empty((height/waves_per_workgroup, width), np.float32)
hHeight2 = np.zeros(shapeNp, np.int32)
hNumActiveTiles = np.empty((1,), np.int32)
hNumBorderTiles = np.empty((1,), np.int32)
hCanDown = np.empty((hDown.shape[0]/32, hDown.shape[1]) , np.uint32)
hCanUp = np.empty((hUp.shape[0]/32, hUp.shape[1]) , np.uint32)
hCanRight = np.empty((hRight.shape[0]/32, hRight.shape[1]) , np.uint32)
hCanLeft = np.empty((hLeft.shape[0]/32, hLeft.shape[1]) , np.uint32)
hNumIterations = np.empty((1,), np.int32)

dImg = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hImg)
dUp = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hUp)
dDown = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hDown)
dLeft = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hLeft)
dTranspose = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hTranspose)
dRight = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hRight)
dSrc = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hSrc)
dSink = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hSink)
dHeight = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hHeight)
dExcess = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hExcess)
dBorder = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hBorder)
dBorderUp = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hBorder)
dBorderDown = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hBorder)
dBorderLeft = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hBorder)
dBorderRight = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hBorder)
dHeight2 = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hHeight2)
dNumActiveTiles = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hNumActiveTiles)
dNumBorderTiles = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hNumActiveTiles)
dCanUp = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hUp)
dCanDown = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hCanDown)
dCanLeft = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hCanLeft)
dCanRight = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hCanRight)
dNumIterations = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hNumIterations)

streamCompact = StreamCompact(context, [device], tilesW*tilesH)

tilesPush = TileFlags(streamCompact)
tilesBfs = TileFlags(streamCompact)
tilesEdges = TileFlags(streamCompact)
tilesBorder = TileFlags(streamCompact)

listBorder = List(streamCompact)
listPush = List(streamCompact)

hWeightMin = 0
hWeightMax = LAMBDA * 1.0/EPSILON


class TileListFilter(ChainedFilter):
	def __init__(self, canvas, tileflags, size):
		ChainedFilter.__init__(self, canvas)

		filename = os.path.join(os.path.dirname(__file__), 'graphcut.cl')
		program = createProgram(canvas.context, canvas.devices, options, filename)

		self.output = cl.Buffer(context, cm.READ_WRITE, size=size)

		self.kern = cl.Kernel(program, 'tilelist')
		filename = os.path.join(os.path.dirname(__file__), 'colorize.cl')
		program = createProgram(canvas.context, canvas.devices, options, filename)

		self.args = [
			None,
			np.array(self.hues, np.int32),
			None,
			None,
			None,
			]

		self.kern = cl.Kernel(program, 'colorize_i32')


		self.canvas = canvas
		self.tileflags = tileflags

	def execute(self, input):
		self.kern(self.canvas.queue, gWorksizeWaves, lWorksizeWaves, input)

		self.argsC[0] = np.array([0, self.tileflags.iteration-1], np.int32)
		self.argsC[2] = input
		self.argsC[3] = np.array(input.shape, np.int32)
		self.argsC[4] = input


		self.output.shape = input.shape

		return self.output
class Transpose(ChainedFilter):
	def __init__(self, canvas, size):
		ChainedFilter.__init__(self, canvas)

		filename = os.path.join(os.path.dirname(__file__), 'graphcut.cl')
		program = createProgram(canvas.context, canvas.devices, options, filename)

		self.output = cl.Buffer(context, cm.READ_WRITE, size=size)

		self.kern = cl.Kernel(program, 'tranpose')

		self.canvas = canvas

	def execute(self, input):
		self.kern(self.canvas.queue, gWorksizeTiles16, lWorksizeTiles16, input, self.output)

		self.output.shape = input.shape

		return self.output

filters = [
	Colorize(canvas, (1, 72), (240, 0),
		formatIn=(cl.Buffer, np.int32),
		formatOut=(cl.Image, cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT8))
	)
]
window.addLayer('height', dHeight, shape, datatype=np.int32, filters=filters)
window.addLayer('height2/bfs', dHeight2, shape, datatype=np.int32, filters=filters)

filters = [
	Colorize(canvas, (0, hWeightMax), (240, 0),
		formatIn=(cl.Buffer, np.float32),
		formatOut=(cl.Image, cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT8))
	)
]
window.addLayer('up', dUp, shape, datatype=np.float32, filters=filters)
window.addLayer('down', dDown, shape, datatype=np.float32, filters=filters)

filters = [
	Transpose(canvas, dLeft.size),
	Colorize(canvas, (0, hWeightMax), (240, 0),
		formatIn=(cl.Buffer, np.float32),
		formatOut=(cl.Image, cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT8))
	)
]
window.addLayer('left', dLeft, shape, datatype=np.float32, filters=filters)
window.addLayer('right', dRight, shape, datatype=np.float32, filters=filters)

filters = []
window.addLayer('img', dImg, shape, datatype=np.int32, filters=filters)

filters = [
	Colorize(canvas, (0.01, 48), (240, 0),
		formatIn=(cl.Buffer, np.float32),
		formatOut=(cl.Image, cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT8))
	)
]
window.addLayer('excess', dExcess, shape, datatype=np.float32, filters=filters)

#vTilesBorderList = window.addView(shapeNp, 'Border List')
#vTilesBorder = window.addView(shapeNp, 'TilesBorder')
#vHeight = window.addViewNp(hHeight, 'height')
#vTilesList = window.addView(shapeNp, 'TilesList')
#vTilesBfs = window.addView(shapeNp, 'TilesBfs')
#vBfs = window.addView(shapeNp, 'bfs')
#vBorder = window.addView(shapeNp, 'Border')
#vEdgesBfs = window.addView(shapeNp, 'EdgesBfs')
#vTilesPush = window.addView(shapeNp, 'TilesPush')
#vActive = window.addView(shapeNp, 'active')


lWorksizeTiles16 = (16, 16)
gWorksizeTiles16 = clutil.roundUp(shape, lWorksizeTiles16)

lWorksizeWaves = (wave_bredth, waves_per_workgroup)
gWorksizeWaves = (width, height/wave_length)

lWorksizeSingleWave = (wave_bredth, 1)
gWorksizeSingleWave = (width, height/(waves_per_workgroup*wave_length))

lWorksizeBorderAdd = (wave_bredth, )

argsInit = [
	listPush.dList,
	np.int32(tilesW),
	np.int32(tilesH),
	dImg,
	dUp,
	dDown,
	dLeft,
	dRight,
	cl.LocalMemory(szInt*(waves_per_workgroup*wave_length+2)*(wave_bredth+2))
]

argsLoadTiles = [
	listBorder.dList,
	np.int32(tilesW),
	np.int32(tilesH),
	dImg,
	dUp,
	dDown,
	dLeft,
	dRight,
	cl.LocalMemory(szInt*(waves_per_workgroup*wave_length+2)*(wave_bredth+2)),
	dSink,
	dSrc,
	dExcess
]

init_bfs_iteration = None

def constructor():
	global build_iteration, bfs_iteration, iteration, border_iteration, init_bfs_iteration

	iteration = 1
	init_bfs_iteration = tilesBfs.iteration

	hDown[:] = 0
	hUp[:] = 0
	hRight[:] = 0
	hLeft[:] = 0
#	hBorder[:] = 0

	cl.enqueue_copy(queue, dDown, hDown).wait()
	cl.enqueue_copy(queue, dUp, hUp).wait()
	cl.enqueue_copy(queue, dRight, hRight).wait()
	cl.enqueue_copy(queue, dLeft, hLeft).wait()
#	cl.enqueue_copy(queue, dBorder, hBorder).wait()

constructor()

def init():
	tilesPush.increment()

	argsInitGC = [
		dSrc,
		dSink,
		cl.LocalMemory(szInt*1),
		tilesPush.dTiles,
		np.int32(tilesPush.iteration),
		dExcess
	]
	kernInitGC(queue, gWorksizeWaves, lWorksizeWaves, *argsInitGC).wait()

	tilesPush.flag(queue)
	listPush.build(queue, tilesPush)
	gWorksizeList = (lWorksizeWaves[0]*listPush.length, lWorksizeWaves[1])

	kernInitNeighbourhood(queue, gWorksizeList, lWorksizeWaves, *argsInit).wait()

	tilesPush.flag(queue)
	listPush.build(queue, tilesPush)

	init_bfs_iteration = tilesBfs.iteration
	startBfs()

	tilesBfs.flag(queue, streamCompact.OPERATOR_GT, init_bfs_iteration)
	listPush.build(queue, tilesBfs)
	activeListAfterBfs()

	tilesPush.flag(queue)
	listPush.build(queue, tilesPush)

	argsPushUpDown[11] = np.int32(tilesBorder.iteration+1)
	argsPushLeftRight[13] = np.int32(tilesBorder.iteration+1)

	window.updateCanvas()



weightRange = (0.001, hWeightMax)
heightRange = (0, MAX_HEIGHT)
excessRange = (0,001, hExcess.max())
borderRange = (0.001, 48)
reversedHue = (240, 0)

def mapBorder():
	kernViewBorder(queue, gWorksizeSingleWave, lWorksizeSingleWave, dBorder, vBorder, np.uint32(1)).wait()
	colorize.colorize(queue, vBorder, val=borderRange, hue=reversedHue)

def mapActive():
	kernViewActive(queue, gWorksizeTiles16, lWorksizeTiles16, dHeight, dExcess, dDown, dRight, dUp, dLeft, vActive).wait()

def mapTilesBfs():
	kernMapActiveTiles(queue, gWorksizeTiles16, lWorksizeTiles16, tilesBfs.dTiles, vTilesBfs, np.int32(tilesBfs.iteration-1)).wait()
	colorize.colorize(queue, vTilesBfs, val=(init_bfs_iteration-1, tilesBfs.iteration-1), hue=reversedHue, typeIn=np.int32)

def mapTilesBorder():
	kernMapActiveTiles(queue, gWorksizeTiles16, lWorksizeTiles16, tilesBorder.dTiles, vTilesBorder, np.int32(tilesBorder.iteration)).wait()
	colorize.colorize(queue, vTilesBorder, val=(0, tilesBorder.iteration), hue=reversedHue, typeIn=np.int32)

def mapEdgesBfs():
	kernMapActiveTiles(queue, gWorksizeTiles16, lWorksizeTiles16, tilesEdges.dTiles, vEdgesBfs, np.int32(tilesBfs.iteration-1)).wait()
	colorize.colorize(queue, vEdgesBfs, val=(0, tilesBfs.iteration-1), hue=reversedHue, typeIn=np.int32)

def mapActiveTiles():
	kernMapActiveTiles(queue, gWorksizeTiles16, lWorksizeTiles16, tilesPush.dTiles, vTilesPush, np.int32(tilesPush.iteration-1)).wait()
	colorize.colorize(queue, vTilesPush, val=(0, tilesPush.iteration-1), hue=reversedHue, typeIn=np.int32)

def mapTileList():
	args = [
		listPush.dList,
		np.int32(tilesW),
		np.int32(tilesPush.iteration-1),
		vTilesList
	]

	if listPush.length == 0:
		return

	gWorksizeList = (lWorksizeWaves[0]*listPush.length, lWorksizeWaves[1])
	kernClear(queue, gWorksizeList, lWorksizeWaves, vTilesList)
	kernMapTileList(queue, gWorksizeList, lWorksizeWaves, *args).wait()

def mapTileBorderList():
	args = [
		listBorder.dList,
		np.int32(tilesW),
		np.int32(tilesBorder.iteration-1),
		vTilesBorderList
	]

	if listBorder.length == 0:
		return

	gWorksizeBorderList = (listBorder.length*wave_bredth, lWorksizeWaves[1])

	kernClear(queue, gWorksizeWaves, lWorksizeWaves, vTilesBorderList)
	kernMapTileList(queue, gWorksizeBorderList, lWorksizeWaves, *args).wait()

#window.setMap(vDown, functools.partial(colorize.colorize, queue, dDown, val=weightRange, hue=reversedHue, dOut=vDown))
#window.setMap(vUp, functools.partial(colorize.colorize, queue, dUp, val=weightRange, hue=reversedHue, dOut=vUp))
#window.setMap(vLeft, functools.partial(map1, dLeft, vLeft))
#window.setMap(vRight, functools.partial(map1, dRight, vRight))
#window.setMap(vExcess, mapExcess)
#window.setMap(vHeight, mapHeight)
#window.setMap(vBfs, mapBfs)
#window.setMap(vBorder, mapBorder)
#window.setMap(vActive, mapActive)
#window.setMap(vTilesList, mapTileList)
#window.setMap(vTilesPush, mapActiveTiles)
#window.setMap(vTilesBorder, mapTilesBorder)
#window.setMap(vTilesBorderList, mapTileBorderList)
#window.setMap(vTilesBfs, mapTilesBfs)
#window.setMap(vEdgesBfs, mapEdgesBfs)

argsInitBfs = [
	listPush.dList,
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
	np.int32(tilesW),
	np.int32(tilesH),
	tilesBfs.dTiles,
	None
	]

def initBfs():
	pass

def intratile_gaps():
	gWorksizeBfs = (lWorksizeWaves[0]*int(listPush.length), lWorksizeWaves[1])

	tilesEdges.increment()

	args = [
		listPush.dList,
		dHeight,
		cl.LocalMemory(szInt*(wave_bredth+2)*(wave_length*waves_per_workgroup+2)),
		dCanDown, dCanUp, dCanRight, dCanLeft,
		np.int32(width/wave_bredth),
		np.int32(width),
		cl.LocalMemory(szInt*1),
		dNumIterations,
		tilesEdges.dTiles,
		np.int32(tilesEdges.iteration)
	]

	kernBfs(queue, gWorksizeBfs, lWorksizeWaves, *args)


def intertile_gaps():
	global bfs_iteration
	lWorksizeBfs = (lWorksizeWaves[0], )
	gWorksizeBfs = (lWorksizeWaves[0]*int(listPush.length), )

	tilesBfs.increment()

	args = [
		listPush.dList,
		dNumActiveTiles,
		cl.LocalMemory(szInt*3),
		dHeight,
		dCanDown, dCanUp, dCanRight, dCanLeft,
		np.int32(tilesW),
		np.int32(width),
		np.int32(tilesBfs.iteration),
		tilesBfs.dTiles
	]

	kernBfsInterTile(queue, gWorksizeBfs, lWorksizeBfs, *args).wait()


def intercept():
	cl.enqueue_copy(queue, hTilesBorder, tilesBorder.dTiles)
	print hTilesBorder
	pass


def startBfs():
	tilesBfs.increment()
	argsInitBfs[20] = np.int32(tilesBfs.iteration)

	argsInitBfs[3] = dHeight

	gWorksize = (lWorksizeWaves[0]*int(listPush.length), lWorksizeWaves[1])
	kernInitBfs(queue, gWorksize, lWorksizeWaves, *argsInitBfs)

	while True:
		tilesBfs.flag(queue)
		listPush.build(queue, tilesBfs)

		if listPush.length > 0:
			intratile_gaps()
		else:
			break;

		tilesEdges.flag(queue)
		listPush.build(queue, tilesEdges)

		if listPush.length > 0:
			intertile_gaps()
		else:
			break;


def activeListAfterBfs():
	gWorksizeBfs = (lWorksizeWaves[0]*int(listPush.length), lWorksizeWaves[1])
	lWorksizeBfs = lWorksizeWaves

	tilesPush.increment()

	args = [
		listPush.dList,
		np.int32(tilesW),
		dExcess,
		dHeight2,
#		dHeight,
		cl.LocalMemory(szInt*1),
		tilesPush.dTiles,
		np.int32(tilesPush.iteration)
	]

	kernInitPush(queue, gWorksizeBfs, lWorksizeBfs, *args).wait()


def relabel():
	global dHeight, dHeight2

	kernRelabel(queue, gWorksizeTiles16, lWorksizeTiles16, dDown, dRight, dUp, dLeft, dExcess, dHeight, dHeight2)

	tmpHeight = dHeight
	dHeight = dHeight2
	dHeight2 = tmpHeight

argsPushUpDown = [
	listPush.dList,
	np.int32(tilesW),
	dDown,
	dUp,
	dHeight,
	dExcess,
	dBorder,
	cl.LocalMemory(szInt*1),
	tilesBorder.dTiles,
	None,
	tilesPush.dTiles,
	None,
]

argsPushLeftRight = [
	listPush.dList,
	np.int32(tilesW),
	dExcess,
	cl.LocalMemory(4*waves_per_workgroup*wave_length*(wave_bredth+1)),
	dHeight,
	cl.LocalMemory(4*waves_per_workgroup*wave_length*(wave_bredth+1)),
	dRight,
	dLeft,
	dBorder,
	cl.LocalMemory(szInt*1),
	tilesBorder.dTiles,
	None,
	tilesPush.dTiles,
	None,
]

argsAddBorder = [
	listBorder.dList,
	np.int32(tilesW),
	dBorder,
	dExcess,
	None,
]


def push():

	#dHeight and dHeight2 swap after BFS
	argsPushUpDown[4] = dHeight
	argsPushLeftRight[4] = dHeight

	gWorksizeList = (lWorksizeWaves[0]*listPush.length, lWorksizeWaves[1])

	tilesBorder.increment()
	argsPushUpDown[9] = np.int32(tilesBorder.iteration)
	argsPushUpDown[6] = dBorderDown
	kernPushDown(queue, gWorksizeList, lWorksizeWaves, *argsPushUpDown).wait()
	tilesBorder.flag(queue)
	listBorder.build(queue, tilesBorder)
	if listBorder.length:
		argsAddBorder[2] = dBorderDown
		argsAddBorder[4] = np.int32(0)
		gWorksizeBorder2 = (lWorksizeWaves[0]*listBorder.length, )
		kernAddBorder(queue, gWorksizeBorder2, lWorksizeBorderAdd, *argsAddBorder).wait()


	tilesBorder.increment()
	argsPushUpDown[9] = np.int32(tilesBorder.iteration)
	argsPushUpDown[6] = dBorderUp
	kernPushUp(queue, gWorksizeList, lWorksizeWaves, *argsPushUpDown).wait()
	tilesBorder.flag(queue)
	listBorder.build(queue, tilesBorder)
	if listBorder.length:
		argsAddBorder[2] = dBorderUp
		argsAddBorder[4] = np.int32(1)
		gWorksizeBorder2 = (lWorksizeWaves[0]*listBorder.length, )
		kernAddBorder(queue, gWorksizeBorder2, lWorksizeBorderAdd, *argsAddBorder).wait()


	tilesBorder.increment()
	argsPushLeftRight[8] = dBorderRight
	argsPushLeftRight[11] = np.int32(tilesBorder.iteration)
	kernPushRight(queue, gWorksizeList, lWorksizeWaves, *argsPushLeftRight).wait()
	tilesBorder.flag(queue)
	listBorder.build(queue, tilesBorder)
	if listBorder.length:
		argsAddBorder[2] = dBorderRight
		argsAddBorder[4] = np.int32(2)
		gWorksizeBorder2 = (lWorksizeWaves[0]*listBorder.length, )
		kernAddBorder(queue, gWorksizeBorder2, lWorksizeBorderAdd, *argsAddBorder).wait()


	tilesBorder.increment()
	argsPushLeftRight[8] = dBorderLeft
	argsPushLeftRight[11] = np.int32(tilesBorder.iteration)
	kernPushLeft(queue, gWorksizeList, lWorksizeWaves, *argsPushLeftRight).wait()
	tilesBorder.flag(queue)
	listBorder.build(queue, tilesBorder)
	if listBorder.length:
		argsAddBorder[2] = dBorderLeft
		argsAddBorder[4] = np.int32(3)
		gWorksizeBorder2 = (lWorksizeWaves[0]*listBorder.length, )
		kernAddBorder(queue, gWorksizeBorder2, lWorksizeBorderAdd, *argsAddBorder).wait()

iteration = 1
globalbfs = 5

hTilesBorder = np.empty((tilesH, tilesW) , np.int32)
hTilesPush = np.empty((tilesH, tilesW) , np.int32)

def next():
	print 'iteration:', iteration
	global iteration, build_iteration, gWorksizeList, init_bfs_iteration

	push()

	if iteration%globalbfs == 0:
		#load additional tiles dynamically if needed
#		tilesBorder.flagLogical(queue, tilesPush.dTiles,
#								StreamCompact.OPERATOR_GTE, StreamCompact.OPERATOR_EQUAL,
#								(tilesBorder.iteration-(globalbfs*4)), -1,
#								StreamCompact.LOGICAL_AND
#		)
#		listBorder.build(queue, tilesBorder)
#		if listBorder.length > 0:
#			gWorksizeList = (lWorksizeWaves[0]*listBorder.length, lWorksizeWaves[1])
#			kernLoadTiles(queue, gWorksizeList, lWorksizeWaves, *argsLoadTiles)

		#push kernels have marked tilesBorder with starting tilesBorder.increment + 5
		tilesBorder.increment()

		tilesBorder.flag(queue)
		listBorder.build(queue, tilesBorder)
		print listBorder.length
		if listBorder.length > 0:
			gWorksizeList = (lWorksizeWaves[0]*listBorder.length, lWorksizeWaves[1])
			kernLoadTiles(queue, gWorksizeList, lWorksizeWaves, *argsLoadTiles)





		#build list of tiles with excess >= 0 and tiles that received overflow
		tilesPush.flagLogical(queue, tilesBorder.dTiles,
			StreamCompact.OPERATOR_GTE, StreamCompact.OPERATOR_EQUAL,
			tilesPush.iteration-1, tilesBorder.iteration,
			StreamCompact.LOGICAL_OR
		)
		listPush.build(queue, tilesPush)
		init_bfs_iteration = tilesBfs.iteration
		startBfs()

		tilesBfs.flag(queue, streamCompact.OPERATOR_GT, init_bfs_iteration)
		listPush.build(queue, tilesBfs)
		activeListAfterBfs()

		tilesPush.flag(queue)
		listPush.build(queue, tilesPush)
		tilesPush.increment() #one more increment for additional tiles flaged in activeListAfterBfs()

		argsPushUpDown[11] = np.int32(tilesBorder.iteration+1)
		argsPushLeftRight[13] = np.int32(tilesBorder.iteration+1)

		print 'active tiles:, ', listPush.length
		if listPush.length == 0:
			timer.stop()
			window.updateCanvas()
	else:
		relabel()

	iteration += 1


	window.updateCanvas()

timer = QtCore.QTimer()
timer.timeout.connect(next)

window.addButton("push", push)
window.addButton("relabel", relabel)
window.addButton("next", next)
window.addButton("start", functools.partial(timer.start, 0))
window.addButton("stop", timer.stop)
window.addButton("intercept", intercept)
window.addButton("init", init)
window.addButton("initBfs", initBfs)
window.addButton("startBfs", startBfs)
window.addButton("bfs_intertile", intertile_gaps)
window.addButton("bfs_intratile", intratile_gaps)

#init()
#for i in xrange(16):
#	next()

init()

#window.show()
#sys.exit(app.exec_())

#for x in xrange(4):
#	next()
#
#push()


window.show()
sys.exit(app.exec_())

