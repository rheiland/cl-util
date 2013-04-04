__author__ = 'Marc de Klerk'

import os, sys

from PyQt4 import QtCore, QtGui, QtOpenGL
import numpy as np

import pyopencl as cl
from pyopencl.tools import get_gl_sharing_context_properties

from clutil import roundUp, padArray2D, createProgram, compareFormat, isFormat

try:
	from OpenGL.GL import *
except ImportError:
	app = QtGui.QApplication(sys.argv)
	QtGui.QMessageBox.critical(None, "Application Error", "error importing PyOpenGL")
	sys.exit(1)
from OpenGL.raw.GL.VERSION.GL_1_5 import glBufferData as rawGlBufferData

cm = cl.mem_flags
szFloat = np.dtype(np.float32).itemsize
szInt = np.dtype(np.int32).itemsize

LWORKGROUP = (16, 16)

class Filter:
	def __init__(self, canvas):
		pass

class CLCanvas(QtOpenGL.QGLWidget):
	class Layer:
		def __init__(self, clobj, pos=None, enabled=True, opacity=1.0, filter=None):
			self.clobj = clobj
			self.opacity = opacity if opacity != None else 1.0
			self.enabled = enabled
			self.pos = pos
			self.filter = filter

	def __init__(self, shape, parent=None):
		super(CLCanvas, self).__init__(parent)

		self.w = 0

		self.pbo = None
		self.tex = None

		self.width = shape[0]
		self.height = shape[1]
		self.shape = shape

		self.zoom = 1.0
		self.transX = 0
		self.transY = 1
		self.flag = 0

		self.viewport = None

		self.resize(self.zoom*self.width, self.zoom*self.height)

		self.initializeGL()
		self.initCL()

		self.installEventFilter(self)

		self.fbo = glGenFramebuffers(1)
		self.rbos = glGenRenderbuffers(2)
		self.rbosCL = [None, None]
		for i in [0, 1]:
			glBindRenderbuffer(GL_RENDERBUFFER, self.rbos[i])
			glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, self.width, self.height)
			self.rbosCL[i] = cl.GLRenderBuffer(self.context, cm.READ_WRITE, int(self.rbos[i]))

		self.rboRead = 0
		self.rboWrite = 1

		self.layers = []

		self.devices = self.context.get_info(cl.context_info.DEVICES)
		self.queue = cl.CommandQueue(self.context, properties=cl.command_queue_properties.PROFILING_ENABLE)

		filename = os.path.join(os.path.dirname(__file__), 'clcanvas.cl')
		program = createProgram(self.context, self.devices, [], filename)

		self.kernBlendImgf = cl.Kernel(program, 'blend_imgf')
		self.kernBlendBufui = cl.Kernel(program, 'blend_bufui')
		self.kernBlendImgui = cl.Kernel(program, 'blend_imgui')

		self.kernFlip = cl.Kernel(program, 'flip')

	def addLayer(self, clobj, shape=None, opacity=None, datatype=None, filter=None):
		if type(clobj) == cl.Image:
			shape = (clobj.get_image_info(cl.image_info.WIDTH), clobj.get_image_info(cl.image_info.HEIGHT))
		elif type(clobj) == cl.Buffer:
			if shape == None:
				raise ValueError('shape required with CL Buffer')

			clobj.shape = shape
			clobj.datatype = datatype

		layer = CLCanvas.Layer(clobj, opacity=opacity, filter=filter)
		self.layers.append(layer)

		return layer

	def initCL(self):
		platforms = cl.get_platforms()

		if sys.platform == "darwin":
			self.context = cl.Context(properties=get_gl_sharing_context_properties(), devices=[])
		else:
			try:
				properties = [(cl.context_properties.PLATFORM, platforms)] + get_gl_sharing_context_properties()
				self.context = cl.Context(properties)
			except:
				raise SystemError('Could not create OpenCL context')

		self.queue = cl.CommandQueue(self.context)

	def paintRenderBuffer(self):
		pass

	def setZoom(self, value):
		self.zoom = value

		self.resize(self.zoom*self.width, self.zoom*self.height)

	def setPbo(self, pbo):
		self.pbo = pbo

	def minimumSizeHint(self):
		return QtCore.QSize(self.zoom*self.width, self.zoom*self.height)

	def initializeGL(self):
		self.makeCurrent()

		glClearColor(0.0, 0.0, 0.0, 0.0)

	def swapRbos(self):
		self.rboRead = not self.rboRead
		self.rboWrite = not self.rboWrite

	def paintEvent(self, event):
		r = event.rect()

		self.transX = -r.x()
		self.transY = -int(self.zoom*self.height - r.height()) + r.y()

		self.viewport = (r.width(), r.height())

		self.rect = event.rect()

		self.makeCurrent()

		self.paintGL()

		if self.doubleBuffer():
			self.swapBuffers()

		glFlush()

	def paintGL(self):
		if len(self.layers) == 0:
			return

		#clear the read  renderbuffer
		glBindRenderbuffer(GL_RENDERBUFFER, self.rbos[self.rboRead])
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.fbo)
		glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, self.rbos[self.rboRead])

		glClear(GL_COLOR_BUFFER_BIT)

		cl.enqueue_acquire_gl_objects(self.queue, self.rbosCL)

		visible = []

		for layer in self.layers:
			if not layer.enabled or layer.opacity == 0:
				continue

			visible.append(layer)

			if layer.opacity == 1.0:
				break

		for layer in reversed(visible):
			args = [
				cl.Sampler(self.context, False, cl.addressing_mode.NONE, cl.filter_mode.NEAREST),
				self.rbosCL[self.rboRead],
				self.rbosCL[self.rboWrite],
				np.float32(layer.opacity),
				layer.clobj
			]

			if layer.filter:
				layer.filter.execute(self.queue, args)
			else:
				gw = roundUp(layer.clobj.shape, LWORKGROUP)

				if isFormat(layer.clobj, (cl.Image, cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT8))):
					self.kernBlendImgf(self.queue, gw, LWORKGROUP, *args)
				if isFormat(layer.clobj, (cl.Image, cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8))):
					self.kernBlendImgui(self.queue, gw, LWORKGROUP, *args)
				elif isFormat(layer.clobj, (cl.Buffer, np.int32)):
					args += [
						np.array(layer.clobj.shape, np.int32),
					]
					self.kernBlendBufui(self.queue, gw, LWORKGROUP, *args)

			self.queue.finish()

			self.swapRbos()

		cl.enqueue_release_gl_objects(self.queue, self.rbosCL)

		gw = roundUp(self.shape, LWORKGROUP)
		args = [
			self.rbosCL[self.rboRead],
			self.rbosCL[self.rboWrite],
			cl.Sampler(self.context, False, cl.addressing_mode.NONE, cl.filter_mode.NEAREST),
		]

		self.kernFlip(self.queue, gw, LWORKGROUP, *args)

		self.queue.finish()

		#Prepare to render into the renderbuffer
		glBindRenderbuffer(GL_RENDERBUFFER, self.rbos[self.rboWrite])
		glBindFramebuffer(GL_READ_FRAMEBUFFER, self.fbo)
		glFramebufferRenderbuffer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, self.rbos[self.rboWrite])

		#Set up to read from the renderbuffer and draw to window-system framebuffer
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
		glViewport(0, 0, self.viewport[0], self.viewport[1])

		#Do the copy
		glBlitFramebuffer(
			int(self.rect.x()/self.zoom),
			self.height - int((self.rect.y() + self.rect.height())/self.zoom),
			int((self.rect.x() + self.rect.width())/self.zoom),
			self.height - int(self.rect.y()/self.zoom),
			0, 0, self.rect.width(), self.rect.height(), GL_COLOR_BUFFER_BIT, GL_NEAREST);

	def eventFilter(self, object, event):
		if hasattr(self, 'mouseDrag') and \
		   (event.type() == QtCore.QEvent.MouseMove and event.buttons() == QtCore.Qt.LeftButton):

			point = (int(event.pos().x()/self.zoom), int(event.pos().y()/self.zoom))

			self.mouseDrag(self.lastMousePos, point)

			self.lastMousePos = point

			return True
		if hasattr(self, 'mousePress') and event.type() == QtCore.QEvent.MouseButtonPress:
			point = (int(event.pos().x()/self.zoom), int(event.pos().y()/self.zoom))

			self.mousePress(point)

			self.lastMousePos = point

			return True

		return False