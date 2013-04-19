__author__ = 'marcdeklerk'

import pyopencl as cl
import numpy as np
import os
from clutil import roundUp, padArray2D, createProgram, Buffer2D

TILEW = 16
TILEH = 16
LWORKGROUP = (16, 16)
DILATE = 2

cm = cl.mem_flags
szFloat = np.dtype(np.float32).itemsize
szInt = np.dtype(np.int32).itemsize
szChar = np.dtype(np.int8).itemsize

TRI_FG_PNG = 0xFF0000FF
TRI_BG_PNG = 0xFFFF0000
TRI_UK_PNG = 0xFF00FF00

TRI_FG = 0
TRI_BG = 1
TRI_UK = 2

class SharedMatting():
	lw = LWORKGROUP

	def __init__(self, context, devices, dImg):
		self.context = context
		self.queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

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
		self.kernProcessTrimap = cl.Kernel(program, 'process_trimap')
		self.trimapFilter = cl.Kernel(program, 'trimap_filter')

		gWorksize = roundUp(self.dim, SharedMatting.lw)
		args = [
			self.dImg,
			self.dLcv,
			self.dAlpha
		]
		self.kernLcv(self.queue, gWorksize, SharedMatting.lw, *args)

	def processTrimap(self, dTriOut, dTriIn, dStrength, threshold):
		elements_per_workitem = 16
		gWorksize = roundUp((self.dim[0]/elements_per_workitem, self.dim[1]/elements_per_workitem), SharedMatting.lw)

		args = [
			dTriOut,
			dTriIn,
			dStrength,
			np.float32(threshold),
			cl.LocalMemory(szChar*(elements_per_workitem*TILEW+2*DILATE)*(elements_per_workitem*TILEH+2*DILATE))
			]

		iterations = 10
		elapsed = 0;
		for i in range(0, iterations):
			event = self.kernProcessTrimap(self.queue, gWorksize, SharedMatting.lw, *args)
			event.wait()
			elapsed += event.profile.end - event.profile.start

		print("Execution time of test: %g ms" % (1e-6*(elapsed)/iterations))

		self.queue.finish()

	def calcMatte(self, dTri):
		gWorksize = roundUp(self.dim, SharedMatting.lw)

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

	def execute(self, queue, args):
		buf = args[-1]
		args.append(np.array(buf.dim, np.int32))

		gw = roundUp(buf.dim, LWORKGROUP)

		self.trimapFilter(queue, gw, LWORKGROUP, *args)

def trimapPngToCharBuf(trimap):
	out = np.empty(trimap.shape, dtype=np.uint8)

	out[trimap == TRI_FG_PNG] = TRI_FG
	out[trimap == TRI_BG_PNG] = TRI_BG
	out[trimap == TRI_UK_PNG] = TRI_UK

	return out

if __name__ == "__main__":
	import Image
	from PyQt4 import QtGui
	from PyQt4 import QtCore, QtGui
	from CLWindow import CLWindow
	from CLCanvas import CLCanvas
	from Colorize import Colorize
	from Brush import Brush
	from GrowCut import GrowCut
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
	hTri = padArray2D(trimapPngToCharBuf(np.array(tri).view(np.uint32).squeeze()), shape, 'edge')

	dImgGC = cl.Image(context,
		cl.mem_flags.READ_ONLY,
		cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8),
		dim
	)
	cl.enqueue_copy(queue, dImgGC, hImg, origin=(0,0), region=dim)

	dImg = Buffer2D(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hImg)
	dTri = Buffer2D(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hTri)

	dStrokes = Buffer2D(context, cm.READ_WRITE, dim, dtype=np.uint8)

	growCut = GrowCut(context, devices, dImgGC, GrowCut.NEIGHBOURHOOD.MOORE, 6, 6, GrowCut.WEIGHT_POW2)

	brushArgs = [
		#		'__write_only image2d_t strokes',
		'__global uchar* strokes',
		'__global uchar* labels_in',
		'__global float* strength_in',
		'__global int* tiles',
		'int iteration',
		'uchar label',
		'int canvasW'
	]
	#	brushCode = 'write_imagef(strokes, gcoord, rgba2f4(label)/255.0f);\n'
	brushCode = 'strokes[gcoord.y*canvasW + gcoord.x] = label;\n'
	brushCode += 'strength_in[gcoord.y*canvasW + gcoord.x] = 1;\n'
	brushCode += 'labels_in[gcoord.y*canvasW + gcoord.x] = label;\n'
	brushCode += 'tiles[(gcoord.y/{0})*{1} + gcoord.x/{2}] = iteration;\n'.format(16, growCut.tilelist.shape[0], 16)

	brush = Brush(context, devices, brushArgs, brushCode)

	label = np.uint32(0xFF0000FF)

	iteration = 0
	refresh = 50

	def next():
		global iteration

		growCut.evolve(queue)

		if growCut.isComplete:
			timer.stop()

#			sm.processTrimap(dTri, growCut.dLabelsOut, growCut.dStrengthIn, 0.985)

#			sm.calcMatte(dTri)

			window.updateCanvas()

		if iteration % refresh == 0:
#			sm.processTrimap(dTri, growCut.dLabelsOut, growCut.dStrengthIn, 0.985)

			window.updateCanvas()

		iteration += 1

	def mouseDrag(pos1, pos2):
		if pos1 == pos2:
			return

		timer.stop()
		brush.draw_gpu(queue, [dStrokes, growCut.dLabelsIn, growCut.dStrengthIn, growCut.tilelist.dTiles, np.int32(growCut.tilelist.iteration), np.int32(label), np.int32(dim[0])], pos1, pos2)
		queue.finish()
#		sm.processTrimap(dTri, growCut.dLabelsOut, growCut.dStrengthIn, 0.985)
		timer.start()

		window.updateCanvas()

	def mousePress(pos):
		timer.stop()
		brush.draw_gpu(queue, [dStrokes, growCut.dLabelsIn, growCut.dStrengthIn, growCut.tilelist.dTiles, np.int32(growCut.tilelist.iteration), np.int32(label), np.int32(dim[0])], pos)
		queue.finish()
#		sm.processTrimap(dTri, growCut.dLabelsOut, growCut.dStrengthIn, 0.985)
		timer.start()

		window.updateCanvas()

	def keyPress(key):
		global label

		if key == QtCore.Qt.Key_B: label = np.uint32(0xFFFF0000)
		elif key == QtCore.Qt.Key_U: label = np.uint32(0xFF00FF00)
		elif key == QtCore.Qt.Key_F: label = np.uint32(0xFF0000FF)

	timer = QtCore.QTimer()
	timer.timeout.connect(next)

	sm = SharedMatting(context, devices, dImg)

	sm.calcMatte(dTri)

	#setup window
	colorize = Colorize(canvas)

#	filter = colorize.factory((Buffer2D, np.int32), (0, 4), Colorize.HUES.STANDARD, (1, 1), (1, 1))
#	window.addLayer('strokes', dStrokes, 0.25, filter=filter)
#	growCut.dLabelsOut.dtype = np.uint32
#	window.addLayer('labels', growCut.dLabelsOut, 1.0)

#	filter = colorize.factory((Buffer2D, np.float32), (0, 1.0), hues=Colorize.HUES.REVERSED)
#	window.addLayer('strength', growCut.dStrengthIn, 1.0, filter=filter)

	window.addLayer('tri', dTri, 1.0, filter=sm)

	filter = colorize.factory((Buffer2D, np.float32), (0, 1.0), (0, 0), (0, 0), (0, 1))
	window.addLayer('alpha', sm.dAlpha, 1.0, filter=filter)

	window.addLayer('fg', sm.dFg)
#	window.addLayer('bg', sm.dBg)

#	filter = colorize.factory((Buffer2D, np.float32), (0, 5000), hues=Colorize.HUES.REVERSED)
#	window.addLayer('lcv', sm.dLcv, 1.0, filter=filter)

	window.addLayer('image', dImg, 0.25)

#	window.setMousePress(mousePress)
#	window.setMouseDrag(mouseDrag)
#	window.setKeyPress(keyPress)

	window.resize(1000, 700)
	window.move(2000, 0)
	window.show()
	sys.exit(app.exec_())
