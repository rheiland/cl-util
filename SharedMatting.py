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

TRI_UK = 0
TRI_FG = 1
TRI_BG = 2

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

#    hTri = padArray2D(trimapPngToCharBuf(np.array(tri).view(np.uint32).squeeze()), shape, 'edge')
        self.dImg = dImg
#        self.dTri = Buffer2D(context, cm.READ_WRITE, self.dim, np.int32)
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
        gWorksize = roundUp((self.dim[0], self.dim[1]), SharedMatting.lw)

        args = [
            dTriOut,
            dTriIn,
            dStrength,
            np.float32(threshold),
            cl.LocalMemory(szChar*(TILEW+2*DILATE)*(TILEH+2*DILATE))
        ]

        self.kernProcessTrimap(self.queue, gWorksize, SharedMatting.lw, *args)

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

    dImgGC = cl.Image(context,
        cl.mem_flags.READ_ONLY,
        cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8),
        dim
    )
    cl.enqueue_copy(queue, dImgGC, hImg, origin=(0,0), region=dim)

    dImg = Buffer2D(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=hImg)

    dStrokes = Buffer2D(context, cm.READ_WRITE, dim, dtype=np.uint8)

    growCut = GrowCut(context, devices, dImgGC, GrowCut.NEIGHBOURHOOD.VON_NEUMANN, GrowCut.WEIGHT_POW2)

    brush = Brush(context, devices, growCut.dLabelsIn)
    brush.setLabel(TRI_FG)

    iteration = 0
    refresh = 100

    def next():
        global iteration

        growCut.evolve(1)
        sm.processTrimap(dStrokes, growCut.dLabelsOut, growCut.dStrengthIn, 0.985)

        if growCut.isComplete:
            window.updateCanvas()
            timer.stop()

#            sm.processTrimap(growCut.dLabelsIn, growCut.dLabelsOut, growCut.dStrengthIn, 0.985)

#            sm.calcMatte(growCut.dLabelsOut)
#            return
#
        if iteration % refresh == 0:
            sm.processTrimap(dStrokes, growCut.dLabelsOut, growCut.dStrengthIn, 0.985)
#
#            sm.calcMatte(growCut.dLabelsOut)
            window.updateCanvas()
#            pass

        iteration += 1

#    sm.processTrimap(dTri, growCut.dLabelsOut, growCut.dStrengthIn, 0.985)


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

        if key == QtCore.Qt.Key_B: brush.setLabel(TRI_BG)
        elif key == QtCore.Qt.Key_U: label = brush.setLabel(TRI_UK)
        elif key == QtCore.Qt.Key_F: label = brush.setLabel(TRI_FG)

    timer = QtCore.QTimer()
    timer.timeout.connect(next)

    sm = SharedMatting(context, devices, dImg)

#    sm.calcMatte(growCut.dLabelsOut)

    #setup window
    colorize = Colorize(canvas)

#    	filter = colorize.factory((Buffer2D, np.int32), (0, 4), Colorize.HUES.STANDARD, (1, 1), (1, 1))
    window.addLayer('strokes', dStrokes, 0.25, filter=sm)
    #	growCut.dLabelsOut.dtype = np.uint32
    #	window.addLayer('labels', growCut.dLabelsOut, 1.0)

    #	filter = colorize.factory((Buffer2D, np.float32), (0, 1.0), hues=Colorize.HUES.REVERSED)
    #	window.addLayer('strength', growCut.dStrengthIn, 1.0, filter=filter)

    window.addLayer('tri', growCut.dLabelsOut, 1.0, filter=sm)
#    window.addLayer('tri', growCut.dLabelsIn, 1.0, filter=sm)

    filter = colorize.factory((Buffer2D, np.float32), (0, 1.0), (0, 0), (0, 0), (0, 1))
#    window.addLayer('alpha', sm.dAlpha, 1.0, filter=filter)

#    window.addLayer('fg', sm.dFg)
    #	window.addLayer('bg', sm.dBg)

    #	filter = colorize.factory((Buffer2D, np.float32), (0, 5000), hues=Colorize.HUES.REVERSED)
    #	window.addLayer('lcv', sm.dLcv, 1.0, filter=filter)

    window.addLayer('image', dImg, 0.25)

    window.setMousePress(mousePress)
    window.setMouseDrag(mouseDrag)
    window.setKeyPress(keyPress)

    window.resize(1000, 700)
    window.move(2000, 0)
    window.show()
    sys.exit(app.exec_())
