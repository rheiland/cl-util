__author__ = 'marcdeklerk'

import sys, os
import numpy as np
import pyopencl as cl
from clutil import createProgram, roundUp
from Buffer2D import Buffer2D
from IncrementalTileList import IncrementalTileList, Operator, Logical
cimport numpy as cnp

DEF szFloat = 4
DEF szInt = 4
DEF szChar = 1
DEF sz32 = 4

cm = cl.mem_flags

DEF WAVE_BREDTH = 32
DEF WAVE_LENGTH = 8
DEF WAVES_PER_WORKGROUP = 4

DEF TILEW = WAVE_BREDTH
DEF TILEH = (WAVES_PER_WORKGROUP * WAVE_LENGTH)

DEF LAMDA_DEFAULT = 60
DEF EPSILON = 0.05
DEF BFS_DEFAULT = 5

cdef class GraphCut:
    lWorksize = (16, 16)
    lWorksizeSingleWave = (WAVE_BREDTH, 1)
    lWorksizeWaves = (WAVE_BREDTH, WAVES_PER_WORKGROUP)
    lWorksizeBorderAdd = (WAVE_BREDTH, )

    cdef:
        readonly float lamda
        readonly float epsilon

        object queue

        object kernInitGC
        object kernLoad_tiles
        object kernPushUp
        object kernPushDown
        object kernPushLeft
        object kernPushRight
        object kernRelabel
        object kernAddBorder
        object kernInitBfs
        object kernBfsIntraTile
        object kernBfsInterTile
        object kernCheckCompletion
        object kernFilterTranspose

        cnp.ndarray hIsCompleted

        readonly object dImg,
        readonly object dUp,
        readonly object dDown,
        readonly object dLeft,
        readonly object dRight,
        readonly object dHeight,
        readonly object dHeight2,
        readonly object dBorder,
        readonly object dCanUp,
        readonly object dCanDown,
        readonly object dCanLeft,
        readonly object dCanRight,
        readonly object dIsCompleted

        tuple gWorksize
        tuple gWorksizeWaves
        tuple gWorksizeSingleWave

        readonly object tilelistLoad
        readonly object tilelistBfs
        readonly object tilelistEdges
        readonly object tilelistBorder

    def __init__(self, context, devices, dImg, lamda=LAMDA_DEFAULT):
        self.lamda = lamda
        self.epsilon = EPSILON

        dim = dImg.dim
        tilesW = dim[0] / TILEW
        tilesH = dim[1] / TILEH

        MAX_HEIGHT = dim[0] * dim[1]

        options = []
        options += ['-D MAX_HEIGHT=' + repr(MAX_HEIGHT)]
        options += ['-D LAMBDA=' + repr(lamda)]
        options += ['-D EPSILON=' + repr(EPSILON)]
        options += ['-D TILESW=' + str(tilesW)]
        options += ['-D TILESH=' + str(tilesH)]

        self.queue = cl.CommandQueue(context,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

        dir = os.path.dirname(__file__)

        program = createProgram(context, devices, options,
            os.path.join(dir, 'graphcut.cl'))
        self.kernInitGC = cl.Kernel(program, 'init_gc')
        self.kernLoad_tiles = cl.Kernel(program, 'load_tiles')
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
        self.kernFilterTranspose = cl.Kernel(program, 'filter_transpose')

        size = sz32 * dim[0] * dim[1]
        sizeSingleWave = sz32 * dim[0] * (dim[1] / TILEH)
        sizeCompressedWave = sz32 * dim[0] * (dim[1] / TILEH)

        self.hIsCompleted = np.array((1, ), np.int32)

        self.dImg = dImg
        self.dUp = Buffer2D(context, cm.READ_WRITE, dim=dim, dtype=np.float32)
        self.dDown = Buffer2D(context, cm.READ_WRITE, dim=dim, dtype=np.float32)
        self.dLeft = Buffer2D(context, cm.READ_WRITE, dim=dim, dtype=np.float32)
        self.dRight = Buffer2D(context, cm.READ_WRITE, dim=dim,
            dtype=np.float32)
        self.dHeight = Buffer2D(context, cm.READ_WRITE, dim=dim, dtype=np.int32)
        self.dHeight2 = Buffer2D(context, cm.READ_WRITE, dim=dim,
            dtype=np.int32)
        self.dBorder = cl.Buffer(context, cm.READ_WRITE, sizeSingleWave)
        self.dCanUp = cl.Buffer(context, cm.READ_WRITE, sizeSingleWave)
        self.dCanDown = cl.Buffer(context, cm.READ_WRITE, sizeSingleWave)
        self.dCanLeft = cl.Buffer(context, cm.READ_WRITE, sizeSingleWave)
        self.dCanRight = cl.Buffer(context, cm.READ_WRITE, sizeSingleWave)
        self.dIsCompleted = cl.Buffer(context, cm.READ_WRITE, szInt)

        shapeTiles = (tilesW, tilesH)

        self.tilelistLoad = IncrementalTileList(context, devices, dim, (32,
                                                                        32))
        self.tilelistBfs = IncrementalTileList(context, devices, dim, (32,
                                                                       32))
        self.tilelistEdges = IncrementalTileList(context, devices, dim,
            (32, 32))
        self.tilelistBorder = IncrementalTileList(context, devices, dim, (32,
                                                                          32))

        self.gWorksize = roundUp(dim, self.lWorksize)
        self.gWorksizeWaves = (dim[0], dim[1] / WAVE_LENGTH)
        self.gWorksizeSingleWave = (
            dim[0], dim[1] / (WAVES_PER_WORKGROUP * WAVE_LENGTH))

    def intratile_gaps(self):
        gWorksizeBfs = (WAVE_BREDTH * self.tilelistBfs.length, WAVE_LENGTH)

        args = [
            self.tilelistBfs.d_list,
            self.dHeight,
            cl.LocalMemory(szInt * (WAVE_BREDTH + 2) * (
                WAVE_LENGTH * WAVES_PER_WORKGROUP + 2)),
            self.dCanDown, self.dCanUp, self.dCanRight, self.dCanLeft,
            cl.LocalMemory(szInt * 1),
            self.tilelistEdges.d_tiles,
            np.int32(self.tilelistEdges.increment())
        ]
        self.kernBfsIntraTile(self.queue, gWorksizeBfs, self.lWorksizeWaves,
            *args).wait()

    def intertile_gaps(self):
        lWorksizeBfs = (WAVE_BREDTH, )
        gWorksizeBfs = (WAVE_BREDTH * self.tilelistEdges.length, )

        args = [
            self.tilelistEdges.d_list,
            cl.LocalMemory(szInt * 3),
            self.dHeight,
            self.dCanDown, self.dCanUp, self.dCanRight, self.dCanLeft,
            np.int32(self.tilelistBfs.increment()),
            self.tilelistBfs.d_tiles
        ]
        self.kernBfsInterTile(self.queue, gWorksizeBfs, lWorksizeBfs,
            *args).wait()

    def startBfs(self, dExcess):
        gWorksize = (
            WAVE_BREDTH * int(self.tilelistLoad.length), WAVES_PER_WORKGROUP)

        args = [
            self.tilelistLoad.d_list,
            dExcess,
            cl.LocalMemory(szInt * TILEH * TILEW),
            self.dHeight,
            cl.LocalMemory(szInt * (2 + 4)),
            self.dDown, self.dUp, self.dRight, self.dLeft,
            self.dCanDown, self.dCanUp, self.dCanRight, self.dCanLeft,
            cl.LocalMemory(szChar * WAVE_BREDTH * WAVES_PER_WORKGROUP),
            cl.LocalMemory(szChar * WAVE_BREDTH * WAVES_PER_WORKGROUP),
            cl.LocalMemory(szChar * WAVE_BREDTH * WAVES_PER_WORKGROUP),
            cl.LocalMemory(szChar * WAVE_BREDTH * WAVES_PER_WORKGROUP),
            self.tilelistBfs.d_tiles,
            np.int32(self.tilelistBfs.increment())
        ]
        self.kernInitBfs(self.queue, gWorksize, self.lWorksizeWaves,
            *args).wait()

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

        self.kernRelabel(self.queue, self.gWorksize,
            GraphCut.lWorksize, *args)

        tmpHeight = self.dHeight
        self.dHeight = self.dHeight2
        self.dHeight2 = tmpHeight

    def push(self, dExcess):
        gWorksize = (
        WAVE_BREDTH * self.tilelistLoad.length, WAVES_PER_WORKGROUP)

        argsPushUpDown = [
            self.tilelistLoad.d_list,
            self.dDown, self.dUp,
            self.dHeight,
            dExcess,
            self.dBorder,
            cl.LocalMemory(szInt * 1),
            self.tilelistBorder.d_tiles,
            None
        ]

        argsPushLeftRight = [
            self.tilelistLoad.d_list,
            dExcess,
            cl.LocalMemory(szFloat * TILEH * (TILEW + 1)),
            self.dHeight,
            cl.LocalMemory(szFloat * TILEH * (TILEW + 1)),
            self.dRight, self.dLeft,
            self.dBorder,
            cl.LocalMemory(szInt * 1),
            self.tilelistBorder.d_tiles,
            None
        ]

        argsAddBorder = [
            self.tilelistBorder.d_list,
            self.dBorder,
            dExcess,
            None,
        ]

        argsPushUpDown[8] = np.int32(self.tilelistBorder.increment())
        self.kernPushDown(self.queue, gWorksize, self.lWorksizeWaves,
            *argsPushUpDown).wait()

        self.tilelistBorder.build()
        if self.tilelistBorder.length:
            argsAddBorder[3] = np.int32(0)
            gWorksizeBorder = (WAVE_BREDTH * self.tilelistBorder.length, )
            self.kernAddBorder(self.queue, gWorksizeBorder,
                self.lWorksizeBorderAdd,
                *argsAddBorder).wait()

        argsPushUpDown[8] = np.int32(self.tilelistBorder.increment())
        self.kernPushUp(self.queue, gWorksize, self.lWorksizeWaves,
            *argsPushUpDown).wait()

        self.tilelistBorder.build()
        if self.tilelistBorder.length:
            argsAddBorder[3] = np.int32(1)
            gWorksizeBorder = (WAVE_BREDTH * self.tilelistBorder.length, )
            self.kernAddBorder(self.queue, gWorksizeBorder,
                self.lWorksizeBorderAdd,
                *argsAddBorder).wait()

        argsPushLeftRight[10] = np.int32(self.tilelistBorder.increment())
        self.kernPushRight(self.queue, gWorksize, self.lWorksizeWaves,
            *argsPushLeftRight).wait()

        self.tilelistBorder.build()
        if self.tilelistBorder.length:
            argsAddBorder[3] = np.int32(2)
            gWorksizeBorder = (WAVE_BREDTH * self.tilelistBorder.length, )
            self.kernAddBorder(self.queue, gWorksizeBorder,
                self.lWorksizeBorderAdd,
                *argsAddBorder).wait()

        argsPushLeftRight[10] = np.int32(self.tilelistBorder.increment())
        self.kernPushLeft(self.queue, gWorksize, self.lWorksizeWaves,
            *argsPushLeftRight).wait()

        self.tilelistBorder.build()
        if self.tilelistBorder.length:
            argsAddBorder[3] = np.int32(3)
            gWorksizeBorder = (WAVE_BREDTH * self.tilelistBorder.length, )
            self.kernAddBorder(self.queue, gWorksizeBorder,
                self.lWorksizeBorderAdd,
                *argsAddBorder).wait()

    def reset(self):
        self.iteration = 1

    def isCompleted(self, dExcess):
        cl.enqueue_copy(self.queue, self.dIsCompleted, np.int32(True))

        gWorksize = (
        WAVE_BREDTH * self.tilelistLoad.length, WAVES_PER_WORKGROUP)

        args = [
            self.tilelistLoad.d_list,
            dExcess,
            self.dHeight,
            cl.LocalMemory(szInt * 1),
            self.dIsCompleted
        ]
        self.kernCheckCompletion(self.queue, gWorksize, self.lWorksizeWaves,
            *args)

        cl.enqueue_copy(self.queue, self.hIsCompleted, self.dIsCompleted).wait()

        return True if self.hIsCompleted[0] else False

    def cut(self, dExcess, bfs=BFS_DEFAULT):
        loadIteration = self.tilelistLoad.iteration

        argsInitGC = [
            self.dUp,
            self.dDown,
            self.dLeft,
            self.dRight,
            dExcess,
            cl.LocalMemory(szInt * 1),
            self.tilelistLoad.d_tiles,
            np.int32(self.tilelistLoad.increment()),
        ]

        argsLoad_tiles = [
            self.tilelistLoad.d_list,
            self.dImg,
            self.dUp, self.dDown, self.dLeft, self.dRight,
            cl.LocalMemory(szInt * (TILEH + 2) * (TILEW + 2))
        ]

        self.kernInitGC(self.queue, self.gWorksizeWaves, self.lWorksizeWaves,
            *argsInitGC).wait()

        self.tilelistLoad.build()
        gWorksize = (
        WAVE_BREDTH * self.tilelistLoad.length, WAVES_PER_WORKGROUP)
        self.kernLoad_tiles(self.queue, gWorksize, self.lWorksizeWaves,
            *argsLoad_tiles).wait()

        self.startBfs(dExcess)

        iteration = 1

        while True:
#            print 'iteration:', iteration

            self.push(dExcess)

            if iteration % bfs == 0:
                self.tilelistLoad.increment()

                self.tilelistLoad.incorporate(self.tilelistBorder.d_tiles,
                    Operator.LTE, loadIteration,
                    Operator.GTE,
                    self.tilelistBorder.iteration - bfs * 4,
                    Logical.AND
                )

                self.tilelistLoad.build()
                if self.tilelistLoad.length > 0:
#                    print '-- loading {0} tiles'.format(
#                        self.tilelistLoad.length)
                    gWorksize = (
                        WAVE_BREDTH * self.tilelistLoad.length,
                        WAVES_PER_WORKGROUP)
                    self.kernLoad_tiles(self.queue, gWorksize,
                        self.lWorksizeWaves,
                        *argsLoad_tiles).wait()

                self.tilelistLoad.build(Operator.GT, loadIteration)
                self.startBfs(dExcess)

                if self.isCompleted(dExcess):
                    return

#                print 'active tiles:, ', self.tilelistLoad.length
            else:
                self.relabel(dExcess)

            iteration += 1

    def reset(self):
        pass

    def execute(self, queue, input):
        context = queue.get_info(cl.command_queue_info.CONTEXT)

        output = Buffer2D(context, cm.READ_WRITE, input.dim, dtype=np.float32)

        args = [
            input,
            np.array(input.dim, np.int32),
            output
        ]

        self.kernFilterTranspose(queue, self.gWorksizeWaves,
            self.lWorksizeWaves, *args)

        return output