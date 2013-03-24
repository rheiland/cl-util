__author__ = 'Marc de Klerk'

import sys

from PyQt4 import QtCore, QtGui, QtOpenGL
import numpy as np

try:
	from OpenGL.GL import *
except ImportError:
	app = QtGui.QApplication(sys.argv)
	QtGui.QMessageBox.critical(None, "Application Error", "error importing PyOpenGL")
	sys.exit(1)
from OpenGL.raw.GL.VERSION.GL_1_5 import glBufferData as rawGlBufferData

szFloat = np.dtype(np.float32).itemsize
szInt = np.dtype(np.int32).itemsize

class GLCanvas(QtOpenGL.QGLWidget):
	class View:
		def __init__(self, shape, pos, enabled=True, opacity=1.0, pbo=False):
			self.opacity = opacity
			self.shape = shape
			self.enabled = enabled
			self.map = None
			self.pos = pos
			self.pbo = False

			if pbo:
				self.pbo = glGenBuffers(1)
				glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo)
				glBufferData(GL_PIXEL_UNPACK_BUFFER, szInt*shape[1]*shape[0], None, GL_STATIC_DRAW)

			self.tex = glGenTextures(1)

			glBindTexture(GL_TEXTURE_2D, self.tex)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, shape[1], shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, None)

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

			glFinish()

	def __init__(self, shape, parent=None):
		super(GLCanvas, self).__init__(parent)

		self.w = 0

		self.pbo = None
		self.tex = None

		self.width = shape[0]
		self.height = shape[1]

		self.zoom = 1.0
		self.transX = 0
		self.transY = 1
		self.flag = 0

		self.viewport = None

		self.resize(self.zoom*self.width, self.zoom*self.height)

		self.initializeGL()

		self.installEventFilter(self)

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

	def paintEvent(self, event):
		r = event.rect()

		self.transX = -r.x()
		self.transY = -(self.zoom*self.height - r.height()) + r.y()

		self.viewport = (r.width(), r.height())

		self.makeCurrent()

		self.paintGL()

		if self.doubleBuffer():
			self.swapBuffers()

		glFlush()

	def paintGL(self):
		glViewport(0, 0, self.viewport[0], self.viewport[1])
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glOrtho(0, self.viewport[0], 0, self.viewport[1], -1, 1)
		glTranslatef(self.transX, self.transY, 0)
		glScalef(self.zoom, self.zoom, 1)
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()

		glClear(GL_COLOR_BUFFER_BIT)

		if len(self.layers) == 0:
			return

		glEnable(GL_TEXTURE_2D)
		glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

		for layer in reversed(self.layers):
			if not layer.enabled or layer.opacity == 0:
				continue

			if layer.pos:
				glPushMatrix()
				glLoadIdentity()
				glTranslatef(layer.pos[1], layer.pos[0], 0)

			glColor4f(1.0, 1.0, 1.0, layer.opacity);

			if layer.pbo:
				glBindBuffer(GL_PIXEL_UNPACK_BUFFER, layer.pbo)
				glBindTexture(GL_TEXTURE_2D, layer.tex)
				glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, layer.shape[1], layer.shape[0], GL_RGBA, GL_UNSIGNED_BYTE, None)
			else:
				glBindTexture(GL_TEXTURE_2D, layer.tex)

			glBegin(GL_QUADS)
			glVertex2i(0, 0)
			glTexCoord2i(0, 0)
			glVertex2i(0, layer.shape[0])
			glTexCoord2i(1, 0)
			glVertex2i(layer.shape[1], layer.shape[0])
			glTexCoord2i(1, 1)
			glVertex2i(layer.shape[1], 0)
			glTexCoord2i(0, 1)
			glEnd()

			if layer.pos:
				glPopMatrix()

		glDisable(GL_BLEND)

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