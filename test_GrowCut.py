__author__ = 'marcdeklerk'

import pyopencl as cl
import functools
import Image
import os, sys
from PyQt4 import QtCore, QtGui
from CLWindow import CLWindow
from CLCanvas import CLCanvas
from Brush import Brush
from Colorize import Colorize
from clutil import roundUp, padArray2D
from GrowCut import GrowCut
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
shapeNP = roundUp(shapeNP, GrowCut.lw)
shapeCL = (shapeNP[1], shapeNP[0])

hImg = padArray2D(np.array(img).view(np.uint32).squeeze(), shapeNP, 'edge')

dImg = Image2D(clContext,
    cl.mem_flags.READ_ONLY,
    cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8),
    shapeCL
)
cl.enqueue_copy(queue, dImg, hImg, origin=(0, 0), region=shapeCL).wait()

dStrokes = Buffer2D(clContext, cm.READ_WRITE, shapeCL, dtype=np.uint8)

brush = Brush(clContext, devices, dStrokes)

growCut = GrowCut(clContext, devices, dImg, GrowCut.NEIGHBOURHOOD.VON_NEUMANN, GrowCut.WEIGHT_DEFAULT)

label = 1

iteration = 0
refresh = 100

def update():
    global iteration

    growCut.evolve(1)

    if growCut.isComplete:
        window.updateCanvas()
        timer.stop()
        return

    if iteration % refresh == 0:
        window.updateCanvas()

    iteration += 1

def mouseDrag(pos1, pos2):
    if pos1 == pos2:
        return

    timer.stop()
    brush.draw(pos1, pos2)
    growCut.label(brush.d_points, brush.n_points, brush.label)

    window.updateCanvas()

    timer.start()

def mousePress(pos):
    mouseDrag(pos, None)

def keyPress(key):
    global label

    if key == QtCore.Qt.Key_1: brush.setLabel(1)
    elif key == QtCore.Qt.Key_2: brush.setLabel(2)
    elif key == QtCore.Qt.Key_3: brush.setLabel(3)
    elif key == QtCore.Qt.Key_4: brush.setLabel(4)
    elif key == QtCore.Qt.Key_5: brush.setLabel(5)
    elif key == QtCore.Qt.Key_6: brush.setLabel(6)
    elif key == QtCore.Qt.Key_7: brush.setLabel(7)
    elif key == QtCore.Qt.Key_8: brush.setLabel(8)

timer = QtCore.QTimer()
timer.timeout.connect(update)

#setup window
filter = Colorize(clContext, (0, 8), Colorize.HUES.STANDARD)
window.addLayer('labels', growCut.dLabelsIn, 0.5, filters=[filter])
window.addLayer('labels', growCut.dLabelsOut, 0.5, filters=[filter])
window.addLayer('strokes', dStrokes, 0.25, filters=[filter])

filter = Colorize(clContext, (0, 1.0), hues=Colorize.HUES.REVERSED)
window.addLayer('strength', growCut.dStrengthIn, 1.0, filters=[filter])
window.addLayer('image', dImg)

window.addLayer('tiles', growCut.tilelist, filters=[growCut.tilelist])

window.addButton("start", functools.partial(timer.start, 0))
window.addButton('next', next)
window.setMousePress(mousePress)
window.setMouseDrag(mouseDrag)
window.setKeyPress(keyPress)

window.resize(1000, 700)
window.show()
sys.exit(app.exec_())