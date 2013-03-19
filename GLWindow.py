__author__ = 'Marc de Klerk'

import sys

from PyQt4 import QtCore, QtGui

from GLCanvas import GLCanvas
import pyopencl as cl
from pyopencl.tools import get_gl_sharing_context_properties
from Colorize import Colorize
import numpy as np



try:
	from OpenGL.GL import *
except ImportError:
	app = QtGui.QApplication(sys.argv)
	QtGui.QMessageBox.critical(None, "Application Error", "error importing PyOpenGL")

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480

cm = cl.mem_flags
szFloat = np.dtype(np.float32).itemsize
szInt = np.dtype(np.int32).itemsize

class GLWindow(QtGui.QMainWindow):
	layers = []

	class CenteredScrollArea(QtGui.QScrollArea):
		def __init__(self, parent=None):
			QtGui.QScrollArea.__init__(self, parent)

		def eventFilter(self, object, event):
			if object == self.widget() and event.type() == QtCore.QEvent.Resize:
				QtGui.QScrollArea.eventFilter(self, object, event)
			else:
				QtGui.QScrollArea.eventFilter(self, object, event)

			return False

	def __init__(self, shape=(DEFAULT_HEIGHT, DEFAULT_WIDTH)):
		super(GLWindow, self).__init__(None)

		width = shape[1]
		height = shape[0]

		self.slider = QtGui.QSlider(QtCore.Qt.Horizontal)
		self.slider.setRange(0, 100)
		self.slider.setSingleStep(1)
		self.slider.setTickInterval(10)
		self.slider.valueChanged.connect(self.sigLayerOpacity)

		self.layerList = QtGui.QListWidget(self)
		self.layerList.setAlternatingRowColors(True)
#		self.layerList.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
		self.layerList.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
		self.layerList.setDropIndicatorShown(True)
		self.layerList.setFocusPolicy(QtCore.Qt.NoFocus)
		self.layerList.installEventFilter(self)
		self.layerList.itemClicked.connect(self.sigLayerClicked)
		self.layerList.itemChanged.connect(self.sigCurrentLayerChanged)

		self.canvas = GLCanvas(width, height)
		self.canvas.layers = self.layers

		self.scrollarea = self.CenteredScrollArea()
		self.scrollarea.setWidget(self.canvas)
		self.scrollarea.setAlignment(QtCore.Qt.AlignCenter)

		layoutButtons = QtGui.QHBoxLayout()
		self.widgetButtons = QtGui.QWidget()
		self.widgetButtons.setLayout(layoutButtons)

		self.sliderZoom = QtGui.QSlider(QtCore.Qt.Horizontal)
		self.sliderZoom.setRange(100, 400)
		self.sliderZoom.setSingleStep(1)
		self.sliderZoom.setTickInterval(100)
		self.sliderZoom.setTickPosition(QtGui.QSlider.TicksBelow)
		self.sliderZoom.valueChanged.connect(self.sigZoom)
		self.sliderZoom.setValue(100)

		layout = QtGui.QGridLayout(self)
		layout.addWidget(self.slider)
		layout.addWidget(self.layerList)
		layout.addWidget(self.widgetButtons)
		layout.addWidget(self.sliderZoom)

		side = QtGui.QWidget()
		side.setLayout(layout)

		splitter = QtGui.QSplitter()
		splitter.addWidget(side)
		splitter.addWidget(self.scrollarea)

		self.setCentralWidget(splitter)

		self.initGL()
		self.initCL()

		self.colorize = Colorize(self.context, self.context.devices)

		self.installEventFilter(self)

	def addButton(self, name, action):
		btn = QtGui.QPushButton(name)
		self.widgetButtons.layout().addWidget(btn)

		btn.clicked.connect(action)

	def initGL(self):
		self.glContext = self.canvas.context()
		self.glContext.makeCurrent()

		glClearColor(0.0, 0.0, 0.0, 0.0)

		glFinish()

	def addViewNp(self, arr, name=None):
		out = self.addView(arr.shape, name)

		cl.enqueue_copy(self.queue, out, arr).wait()

		return out

	def setMap(self, view, func):
		layer = self.getLayer(view)

		layer.map = func

	def getLayer(self, dBuf):
		for layer in self.layers:
			if dBuf == layer.dBuf:
				return layer

		return None

	def addView(self, shape, name=None):
		width = shape[1]
		height = shape[0]

		pbo = glGenBuffers(1)
		tex = glGenTextures(1)

		#glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo)
		glBufferData(GL_PIXEL_UNPACK_BUFFER, szInt*width*height, None, GL_STATIC_DRAW)

		glBindTexture(GL_TEXTURE_2D, tex)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

		glFinish()

		view = cl.GLBuffer(self.context, cl.mem_flags.READ_ONLY, int(pbo))

		pos = (self.canvas.height-shape[0], self.canvas.width-shape[1])

		layer = GLCanvas.View(pbo, tex, view, shape, pos)
		self.layers.append(layer)

		item = QtGui.QListWidgetItem(name)
		item.setText(name)
		item.setFlags(	QtCore.Qt.ItemIsUserCheckable |
						QtCore.Qt.ItemIsDragEnabled |
						QtCore.Qt.ItemIsSelectable |
						QtCore.Qt.ItemIsEnabled)
		item.setCheckState(QtCore.Qt.Unchecked)
		item.setData(QtCore.Qt.UserRole, layer)

		self.layerList.addItem(item)

		return view

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

	def sigCurrentLayerChanged(self, item):
		layer = item.data(QtCore.Qt.UserRole).toPyObject()

	def sigLayerOpacity(self, value):
		if self.layerList.currentItem() == None:
			return

		item = self.layerList.currentItem().data(QtCore.Qt.UserRole).toPyObject()
		item.opacity = float(value)/100

		self.updateCanvas()

	def sigZoom(self, value):
		self.setZoom(value)

	def sigLayerClicked(self, item):
		checked = item.checkState() == QtCore.Qt.Unchecked
		layer = item.data(QtCore.Qt.UserRole).toPyObject()

		if layer.enabled == checked:
			layer.enabled = not layer.enabled
			self.updateCanvas()
		else:
			self.slider.setValue(100*layer.opacity)

	def sigLayerIndexesMoved(self):
		del self.layers[:]
		for i in range(self.layerList.count()):
			layer = self.layerList.item(i).data(QtCore.Qt.UserRole).toPyObject()
			self.layers.append(layer)

		self.updateCanvas()

	def updateCanvas(self):
		for layer in self.layers:
			if not layer.enabled or layer.opacity == 0 or layer.map == None:
				continue

			layer.map()

		self.queue.finish()
		self.canvas.repaint()

	def setCursor(self, cursor):
		image = QtGui.QImage(cursor.tostring(), cursor.size[0], cursor.size[1], QtGui.QImage.Format_ARGB32)
		pixmap = QtGui.QPixmap(image)
		cursor = QtGui.QCursor(pixmap)

		self.canvas.setCursor(cursor)

	def setZoom(self, zoom):
		self.canvas.setZoom(float(zoom)/100)

	def setMousePress(self, func):
		self.canvas.mousePress = func

	def setKeyPress(self, func):
		self.keyPress = func

	def eventFilter(self, object, event):
		if hasattr(self, 'keyPress') and event.type() == QtCore.QEvent.KeyPress:
			self.keyPress(event.key())

			return True

		return False