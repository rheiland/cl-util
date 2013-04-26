#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: wraparound=False

#from Tool cimport Tool
import Image
import ImageDraw
import numpy as np
cimport numpy as cnp
import os
from skimage.draw import line, circle

import pyopencl as cl
from clutil import roundUp, Buffer2D, createProgram

cm = cl.mem_flags
cdef int szFloat = np.dtype(np.float32).itemsize
cdef int szInt = np.dtype(np.int32).itemsize

try:
    from OpenGL.GL import *
except ImportError:
    raise ImportError("Error importing PyOpenGL")

LWORKGROUP = (256, )

def argb2abgr(c):
    a = (c >> 24) & 0xFF
    r = (c >> 16) & 0xFF
    g = (c >> 8 ) & 0xFF
    b = (c      ) & 0xFF

    return a << 24 | b << 16 | g << 8 | r

cdef int DEFAULT_RADIUS = 10
cdef int DEFAULT_LABEL = 0

#cdef class Brush(Tool):
cdef class Brush:
    cdef int[2] dim
    cdef int radius
    cdef readonly  int label

    cdef readonly int n_points
    cdef cnp.ndarray h_points

    cdef object queue

    cdef object d_img
    cdef readonly  object d_points

    cdef object kernDraw

    def __init__(self, context, devices, d_img):
#        Tool.__init__(self)
        self.queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

        self.radius = DEFAULT_RADIUS
        self.label = DEFAULT_LABEL

        if type(d_img) == Buffer2D:
            self.d_img = d_img

            self.dim[0] = d_img.dim[0]
            self.dim[1] = d_img.dim[1]

        elif type(d_img) == cl.Image:
            self.dImg = d_img

            width = d_img.get_image_info(cl.image_info.WIDTH)
            height = d_img.get_image_info(cl.image_info.HEIGHT)

            self.dim[0] = width
            self.dim[1] = height
        else:
            raise NotImplementedError('GL Texture')

        self.h_points = np.empty((self.dim[0]*self.dim[1]/4, ), dtype=np.int32)

        self.d_points = cl.Buffer(context, cm.READ_WRITE, szInt*self.dim[0]*self.dim[1]/4)

        options = [
            '-D IMAGEW=' + str(self.dim[0]),
            '-D IMAGEH=' + str(self.dim[1])
        ]

        filename = os.path.join(os.path.dirname(__file__), 'brush.cl')
        program = createProgram(context, devices, options, filename)

        self.kernDraw = cl.Kernel(program, 'draw')

    def getCursor(self):
        shape = (self.diameter, self.diameter)
        cursor = Image.new('RGBA', shape)
        draw = ImageDraw.Draw(cursor)

        color = self.colorCursor[self.type]

        draw.ellipse((0, 0, self.diameter - 1, self.diameter - 1), fill=color)

        return cursor

        def setType(self, type):
            self.type = type;

    @property
    def diameter(self):
        return 2 * self.radius + 1

    def getRadius(self):
        return self.radius

    def setRadius(self, radius, increase=None):
        if increase == '+':
            self.radius += radius
        elif increase == '-':
            self.radius -= radius
        else:
            self.radius = radius

        if radius <= 0:
            self.radius = 1

    def setLabel(self, int label):
        self.label = label

    def draw(self, p0, p1):
        self.points(p0, p1)

        cl.enqueue_copy(self.queue, self.d_points, self.h_points).wait()

#        gWorksize = roundUp((self.n_points, ), LWORKGROUP)

#        self.kernDraw(self.queue, gWorksize, LWORKGROUP, self.d_img, self.d_points, np.int32(self.n_points)).wait()

    cdef cnp.ndarray points(self, p0, p1):
        cdef set points = set()
        cdef cnp.ndarray[cnp.intp_t, ndim=1, mode="c"] xs, ys, xs2, ys2
        cdef int i, j

        if p1 == None:
            ys, xs = circle(p0[1], p0[0], 10)

            for i in range(len(xs)):
                points.add(ys[i]*self.dim[0] + xs[i])
        else:
            ys, xs = line(p0[1], p0[0], p1[1], p1[0])
            ys2, xs2 = circle(10, 10, 10)

            for i in range(len(xs)):
                for j in range(len(xs2)):
                    points.add((ys[i] + ys2[j])*self.dim[0] + xs[i] + xs2[j])

        self.n_points = len(points)

        for i in range(self.n_points):
            self.h_points[i] = points.pop()

            i += 1
