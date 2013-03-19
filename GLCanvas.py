__author__ = 'Marc de Klerk'

import sys

from PyQt4 import QtCore, QtGui, QtOpenGL

try:
	from OpenGL.GL import *
except ImportError:
	app = QtGui.QApplication(sys.argv)
	QtGui.QMessageBox.critical(None, "Application Error", "error importing PyOpenGL")
	sys.exit(1)
from OpenGL.raw.GL.VERSION.GL_1_5 import glBufferData as rawGlBufferData

class GLCanvas(QtOpenGL.QGLWidget):
	def __init__(self, width, height, parent=None):
		super(GLCanvas, self).__init__(parent)

		self.w = 0

		self.pbo = None
		self.tex = None

		self.width = width
		self.height = height

		self.zoom = 1.0
		self.transX = 0
		self.transY = 1
		self.flag = 0

		self.viewW = width
		self.viewH = height

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
		pass

	def paintEvent(self, event):
		r = event.rect()

		self.transX = -r.x()
		self.transY = -(self.zoom*self.height - r.height()) + r.y()

		self.viewW = r.width()
		self.viewH = r.height()

		self.makeCurrent()

		self.paintGL()

		if self.doubleBuffer():
			self.swapBuffers()

		glFlush()

	def paintGL(self):
		glViewport(0, 0, self.viewW, self.viewH)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glOrtho(0, self.viewW, 0, self.viewH, -1, 1)
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

			glColor4f(1.0, 1.0, 1.0, layer.opacity);

			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, layer.pbo)
			glBindTexture(GL_TEXTURE_2D, layer.tex)
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE, None)

			glBegin(GL_QUADS)
			glVertex2i(0,0)
			glTexCoord2i(0,0)
			glVertex2i(0, self.height)
			glTexCoord2i(1,0)
			glVertex2i(self.width, self.height)
			glTexCoord2i(1,1)
			glVertex2i(self.width, 0)
			glTexCoord2i(0,1)
			glEnd()

		glDisable(GL_BLEND)

	def resizeGL(self, width, height):
		self.initializeGL()

		self.paintGL()

	def eventFilter(self, object, event):
		if hasattr(self, 'mouseDrag') and event.type() == QtCore.QEvent.MouseButtonRelease:

			point = (event.pos().x()/self.zoom, event.pos().y()/self.zoom)

			self.mouseDrag(self.lastMousePos, point)

			return True
		if hasattr(self, 'mousePress') and event.type() == QtCore.QEvent.MouseButtonPress:
			point = (event.pos().x()/self.zoom, event.pos().y()/self.zoom)

			self.lastMousePos = point

			self.mousePress(point)

			return True

		return False