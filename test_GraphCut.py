from PyQt4 import QtCore, QtGui
from CLWindow import CLWindow
from CLCanvas import CLCanvas
from Colorize import Colorize
import functools
from GraphCut import GraphCut
import Image
import os, sys
from clutil import roundUp, createProgram, padArray2D
from Buffer2D import Buffer2D
import numpy as np
import pyopencl as cl
import time

cm = cl.mem_flags

img = Image.open("/Users/marcdeklerk/msc/code/dataset/processed/source/800x600/GT04.png")
if img.mode != 'RGBA':
    img = img.convert('RGBA')

shape = (img.size[1], img.size[0])
hImg = padArray2D(np.array(img).view(np.uint32).squeeze(), roundUp(shape, GraphCut.lWorksizeTiles16), 'edge')

width = hImg.shape[1]
height = hImg.shape[0]
dim = (width, height)
shape = (height, width)

app = QtGui.QApplication(sys.argv)
canvas = CLCanvas(dim)
window = CLWindow(canvas)

context = canvas.context

dImg = Buffer2D(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hImg)

devices = context.get_info(cl.context_info.DEVICES)

gc = GraphCut(context, devices, dImg)

window.resize(1000, 700)

hSrc = np.load('scoreFg.npy').reshape(shape)
hSink = np.load('scoreBg.npy').reshape(shape)

dExcess = Buffer2D(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=(hSink-hSrc))

hWeightMin = 0
hWeightMax = gc.lamda * 1.0/GraphCut.EPSILON

filter = Colorize(context, (0, 50), hues=Colorize.HUES.REVERSED)
window.addLayer('excess', dExcess, filters=[filter])

filter = Colorize(context, (1, 144), hues=Colorize.HUES.REVERSED)
window.addLayer('height', gc.dHeight, filters=[filter])
window.addLayer('height2/bfs', gc.dHeight2, filters=[filter])
window.addLayer('tiles Bfs', gc.tilelistBfs, filters=[gc.tilelistBfs])
window.addLayer('tiles Load', gc.tilelistLoad, filters=[gc.tilelistLoad])
window.addLayer('tiles Border', gc.tilelistBorder, filters=[gc.tilelistBorder])
window.addLayer('tiles Edges', gc.tilelistEdges, filters=[gc.tilelistEdges])
window.addLayer('img', dImg)

filter = Colorize(context, (0.001, hWeightMax), hues=Colorize.HUES.REVERSED)
window.addLayer('up', gc.dUp, filters=[filter])
window.addLayer('down', gc.dDown, filters=[filter])

window.addLayer('left', gc.dLeft, filters=[gc, filter])
window.addLayer('right', gc.dRight, filters=[gc, filter])
#
timer = QtCore.QTimer()
#	timer.timeout.connect(next)

def reset():
    cl.enqueue_copy(gc.queue, dExcess, (hSink-hSrc)).wait()

def cut():
    gc.cut(dExcess, 5)
    window.update()

window.addButton("push", gc.push)
window.addButton("relabel", gc.relabel)
window.addButton("cut", cut)
#	window.addButton("start", functools.partial(timer.start, 0))
#	window.addButton("stop", timer.stop)
window.addButton("reset", reset)

window.show()

iterations = 10

elapsed = 0
t1 = t2 = 0
for i in xrange(iterations):
    reset()

    t1 = time.time()

    gc.cut(dExcess, 5)

    elapsed += time.time()-t1
print 'ave time per iteration: ', elapsed/iterations

sys.exit(app.exec_())
