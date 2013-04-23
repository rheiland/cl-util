__author__ = 'Marc de Klerk'

import pyopencl as cl
import numpy as np
import os, sys
from clutil import roundUp, padArray2D, createProgram, Buffer2D
from StreamCompact import StreamCompact, IncrementalTileList

LWORKGROUP = (16, 16)
LWORKGROUP_1D = (256, )

cm = cl.mem_flags
szFloat = np.dtype(np.float32).itemsize
szInt = np.dtype(np.int32).itemsize

TILEW = 16
TILEH = 16

WEIGHT = '(1.0f - X/1.7320508075688772f)'

class GrowCut():
    lw = LWORKGROUP

    class NEIGHBOURHOOD:
        VON_NEUMANN = 0

    lWorksizeTiles16 = (16, 16)

    WEIGHT_DEFAULT = '(1.0f-X/1.7320508075688772f)'
    WEIGHT_POW2 = '(1.0f-pown(X/1.7320508075688772f,2))'
    WEIGHT_POW3 = '(1.0f-pown(X/1.7320508075688772f,3))'
    WEIGHT_POW1_5 = '(1.0f-pow(X/1.7320508075688772f,1.5))'
    WEIGHT_POW_SQRT = '(1.0f-sqrt(X/1.7320508075688772f))'

    def __init__(self, context, devices, img, neighbourhood=NEIGHBOURHOOD.VON_NEUMANN, weight=None):
        self.context = context
        self.queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

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

            dim = (width, height)

        shapeTiles = (dim[0] / TILEW, dim[1] / TILEH)

        streamCompact = StreamCompact(context, devices, shapeTiles[0] * shapeTiles[1])

        self.tilelist = IncrementalTileList(context, devices, shapeTiles)

        self.hHasConverged = np.empty((1,), np.int32)
        self.hHasConverged[0] = False

        self.dLabelsIn = Buffer2D(context, cm.READ_WRITE, dim, np.uint8)
        self.dLabelsOut = Buffer2D(context, cm.READ_WRITE, dim, np.uint8)
        self.dStrengthIn = Buffer2D(context, cm.READ_WRITE, dim, np.float32)
        self.dStrengthOut = Buffer2D(context, cm.READ_WRITE, dim, np.float32)
        self.dHasConverged = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=self.hHasConverged)

        self.args = [
            self.tilelist.dList,
            self.dLabelsIn,
            self.dLabelsOut,
            self.dStrengthIn,
            self.dStrengthOut,
            self.dHasConverged,
            np.int32(self.tilelist.iteration),
            self.tilelist.dTiles,
            cl.LocalMemory(szInt * 9),
            cl.LocalMemory(szInt * (TILEW + 2) * (TILEH + 2)),
            cl.LocalMemory(szFloat * (TILEW + 2) * (TILEH + 2)),
            #			cl.LocalMemory(4*szFloat*(TILEW+2)*(TILEH+2)),
            self.dImg,
            cl.Sampler(context, False, cl.addressing_mode.NONE, cl.filter_mode.NEAREST)
        ]

        self.gWorksize = roundUp(dim, self.lw)
        self.gWorksizeTiles16 = roundUp(dim, self.lWorksizeTiles16)

        options = [
            '-D TILESW=' + str(shapeTiles[0]),
            '-D TILESH=' + str(shapeTiles[1]),
            '-D IMAGEW=' + str(dim[0]),
            '-D IMAGEH=' + str(dim[1]),
            '-D TILEW=' + str(TILEW),
            '-D TILEH=' + str(TILEH),
            '-D G_NORM(X)=' + weight
        ]

        filename = os.path.join(os.path.dirname(__file__), 'growcut.cl')
        program = createProgram(context, devices, options, filename)

        if neighbourhood == GrowCut.NEIGHBOURHOOD.VON_NEUMANN:
            self.kernEvolve = cl.Kernel(program, 'evolveVonNeumann')
        elif neighbourhood == GrowCut.NEIGHBOURHOOD.MOORE:
            self.kernEvolve = cl.Kernel(program, 'evolveMoore')

        self.kernLabel = cl.Kernel(program, 'label')

        self.isComplete = False

    def label(self, d_points, n_points, label):
        gWorksize = roundUp((n_points, ), LWORKGROUP_1D)

        args = [
            self.dLabelsIn,
            self.dStrengthIn,
            d_points,
            np.uint8(label),
            np.int32(n_points),
            self.tilelist.dTiles,
            np.int32(self.tilelist.iteration)
        ]

        self.kernLabel(self.queue, gWorksize, LWORKGROUP_1D, *args).wait()


    def evolve(self, iterations=sys.maxint):
        self.isComplete = False

        self.tilelist.build()

        if self.tilelist.length == 0:
            self.isComplete = True
            return

        self.tilelist.increment()

        gWorksize = (TILEW * self.tilelist.length, TILEH)

        self.args[1] = self.dLabelsIn
        self.args[2] = self.dLabelsOut
        self.args[3] = self.dStrengthIn
        self.args[4] = self.dStrengthOut
        self.args[6] = np.int32(self.tilelist.iteration)

        self.kernEvolve(self.queue, gWorksize, self.lWorksizeTiles16, *self.args).wait()

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
    from PyQt4 import QtCore, QtGui
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
    cl.enqueue_copy(queue, dImg, hImg, origin=(0, 0), region=shapeCL)

    dStrokes = Buffer2D(clContext, cm.READ_WRITE, shapeCL, dtype=np.uint8)

    from Brush import Brush
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

        if key == QtCore.Qt.Key_1: brush.setLabel(0)
        elif key == QtCore.Qt.Key_2: brush.setLabel(1)
        elif key == QtCore.Qt.Key_3: brush.setLabel(2)

    timer = QtCore.QTimer()
    timer.timeout.connect(update)

    colorize = Colorize(canvas)

    #setup window
    filter = colorize.factory((Buffer2D, np.uint8), (0, 3), Colorize.HUES.STANDARD, (1, 1), (1, 1))
#    window.addLayer('labels', growCut.dLabelsIn, 0.5, filter=filter)
    window.addLayer('labels', growCut.dLabelsOut, 0.5, filter=filter)
    window.addLayer('strokes', dStrokes, 0.25, filter=filter)
    window.addLayer('image', dImg)

    filter = colorize.factory((Buffer2D, np.float32), (0, 1.0), hues=Colorize.HUES.REVERSED)
    window.addLayer('strength', growCut.dStrengthIn, 1.0, filter=filter)

    options = [
        '-D IMAGEW={0}'.format(shapeCL[0]),
        '-D IMAGEH={0}'.format(shapeCL[1]),
        '-D TILESW=' + str(growCut.tilelist.shape[0]),
        '-D TILESH=' + str(growCut.tilelist.shape[1])
    ]

    filename = os.path.join(os.path.dirname(__file__), 'graphcut_filter.cl')
    program = createProgram(canvas.context, canvas.devices, options, filename)

    kernTileList = cl.Kernel(program, 'tilelist_growcut')

    class TileListFilter():
        def execute(self, queue, args):
            buf = args[-1]
            args.append(np.array(buf.dim, np.int32))

            args += [
                np.array([growCut.tilelist.iteration - 1, growCut.tilelist.iteration], np.float32),
                np.array(Colorize.HUES.REVERSED, np.float32),
                np.array([1, 1], np.float32),
                np.array([1, 1], np.float32),
                ]
            kernTileList(queue, growCut.gWorksizeTiles16, growCut.lWorksizeTiles16, *args)

    tilelistfilter = TileListFilter()

    window.addLayer('tiles', growCut.tilelist.dTiles, filter=tilelistfilter)

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