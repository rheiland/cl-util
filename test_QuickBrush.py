__author__ = 'marcdeklerk'

import pyopencl as cl
import functools
import Image
import os, sys
from PyQt4 import QtCore, QtGui
from CLWindow import CLWindow
from CLCanvas import CLCanvas
from Colorize import Colorize
from clutil import roundUp, padArray2D
from QuickBrush import QuickBrush
from Buffer2D import Buffer2D
from Image2D import Image2D
import numpy as np
import pyopencl as cl
cm = cl.mem_flags

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
shapeNP = roundUp(shapeNP, QuickBrush.lWorksize)
shapeCL = (shapeNP[1], shapeNP[0])

hImg = padArray2D(np.array(img).view(np.uint32).squeeze(), shapeNP, 'edge')

dImg = Image2D(clContext,
    cl.mem_flags.READ_ONLY,
    cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8),
    shapeCL
)
cl.enqueue_copy(queue, dImg, hImg, origin=(0, 0), region=shapeCL).wait()
dBuf = Buffer2D(clContext, cm.READ_ONLY | cm.COPY_HOST_PTR, hostbuf=hImg)

dStrokes = Buffer2D(clContext, cm.READ_WRITE, shapeCL, dtype=np.uint8)

brush = QuickBrush(clContext, devices, dImg, dStrokes)
brush.setRadius(20)

label = 1

def mouseDrag(pos1, pos2):
    if pos1 == pos2:
        return

    brush.draw(pos1, pos2)

    window.updateCanvas()

def mousePress(pos):
    mouseDrag(pos, None)

def keyPress(key):
    global label

    if key == QtCore.Qt.Key_1: brush.setLabel(1)
    elif key == QtCore.Qt.Key_2: brush.setLabel(2)

#setup window
filter = Colorize(clContext, (0, 200.0), hues=Colorize.HUES.REVERSED)
#window.addLayer('gmm', brush.dScoreBg, 0.5, filters=[filter])
window.addLayer('gmm', brush.dScoreFg, 0.5, filters=[filter])

filter = Colorize(clContext, (0, 2), Colorize.HUES.STANDARD)
window.addLayer('strokes', dStrokes, 0.25, filters=[filter])

window.addLayer('image', dImg)

window.setMousePress(mousePress)
window.setMouseDrag(mouseDrag)
window.setKeyPress(keyPress)

window.resize(1000, 700)
window.show()
sys.exit(app.exec_())