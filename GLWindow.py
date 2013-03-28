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
	raise ImportError("Error importing PyOpenGL")

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

			return True

	def __init__(self, shape=(DEFAULT_WIDTH, DEFAULT_HEIGHT)):
		super(GLWindow, self).__init__(None)

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

		self.canvas = GLCanvas(shape)
		self.canvas.layers = self.layers
		self.canvas.mousePress = self.mousePress
		self.canvas.mouseDrag = self.mouseDrag

		self.scrollarea = QtGui.QScrollArea()
		#self.scrollarea = self.CenteredScrollArea()
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

		layout = QtGui.QGridLayout()
		layout.addWidget(self.slider)
		layout.addWidget(self.layerList)
		layout.addWidget(self.widgetButtons)
		layout.addWidget(self.sliderZoom)

		side = QtGui.QWidget()
		side.setLayout(layout)

		splitter = QtGui.QSplitter()
		splitter.addWidget(side)
		splitter.addWidget(self.scrollarea)
		splitter.setSizes([100, shape[0] + 25])

		self.setCentralWidget(splitter)

		self.initGL()
		self.initCL()

		self.colorize = Colorize(self.clContext, self.clContext.devices)

		self.installEventFilter(self)
		self.canvas.installEventFilter(self)

		self.resize(100 + shape[0] + 24, shape[1] + 10)

	def addButton(self, name, action):
		btn = QtGui.QPushButton(name)
		self.widgetButtons.layout().addWidget(btn)

		btn.clicked.connect(action)

	def initGL(self):
		self.glContext = self.canvas.context()

	def addViewNp(self, arr, name=None, mem_flags=cm.READ_ONLY, buffer=False):
		view = self.addView(arr.shape, name, mem_flags, buffer)

		if buffer:
			cl.enqueue_copy(self.queue, view, arr).wait()
		else:
			cl.enqueue_copy(self.queue, view, arr, origin=(0, 0), region=(arr.shape[1], arr.shape[0])).wait()

		return view

	def setLayerMap(self, name, func):
		item = self.layerList.findItems(name, QtCore.Qt.MatchExactly)[0]

		item.map = func

	def setLayerOpacity(self, name, opacity):
		item = self.layerList.findItems(name, QtCore.Qt.MatchExactly)[0]

		item.view.opacity = opacity

		self.updateCanvas()

	def addView(self, shape, name=None, mem_flags=cm.READ_ONLY, buffer=False):
		pos = (self.canvas.height-shape[0], self.canvas.width-shape[1])
		view = GLCanvas.View(shape, pos, pbo=buffer)

		self.layers.append(view)

		item = QtGui.QListWidgetItem(name)
		item.setText(name)
		item.setFlags(	QtCore.Qt.ItemIsUserCheckable |
						QtCore.Qt.ItemIsDragEnabled |
						QtCore.Qt.ItemIsSelectable |
						QtCore.Qt.ItemIsEnabled)
		item.setCheckState(QtCore.Qt.Checked)
		item.view = view
		item.map = None

		self.layerList.addItem(item)

		if buffer:
			return cl.GLBuffer(self.clContext, mem_flags, int(view.pbo))
		else:
			return cl.GLTexture(self.clContext, mem_flags, GL_TEXTURE_2D, 0, int(view.tex), 2)

	def initCL(self):
		platforms = cl.get_platforms()

		if sys.platform == "darwin":
			self.clContext = cl.Context(properties=get_gl_sharing_context_properties(), devices=[])
		else:
			try:
				properties = [(cl.context_properties.PLATFORM, platforms)] + get_gl_sharing_context_properties()
				self.clContext = cl.Context(properties)
			except:
				raise SystemError('Could not create OpenCL context')

		self.queue = cl.CommandQueue(self.clContext)

	def sigCurrentLayerChanged(self, item):
		layer = item.view

	def sigLayerOpacity(self, value):
		if self.layerList.currentItem() == None:
			return

		item = self.layerList.currentItem().view
		item.opacity = float(value)/100

		self.updateCanvas()

	def sigZoom(self, value):
		self.setZoom(value)

	def sigLayerClicked(self, item):
		checked = item.checkState() == QtCore.Qt.Unchecked
		layer = item.view

		if layer.enabled == checked:
			layer.enabled = not layer.enabled
			self.updateCanvas()
		else:
			self.slider.setValue(100*layer.opacity)

	def sigLayerIndexesMoved(self):
		del self.layers[:]
		for i in range(self.layerList.count()):
			layer = self.layerList.item(i).view
			self.layers.append(layer)

		self.updateCanvas()

	def updateCanvas(self):
		self.canvas.repaint()

	def setCursor(self, cursor):
		image = QtGui.QImage(cursor.tostring(), cursor.size[0], cursor.size[1], QtGui.QImage.Format_ARGB32)
		pixmap = QtGui.QPixmap(image)
		cursor = QtGui.QCursor(pixmap)

		self.canvas.setCursor(cursor)

	def setZoom(self, zoom):
		self.canvas.setZoom(float(zoom)/100)

	def setMousePress(self, func):
		self.userMousePress = func

	def setMouseDrag(self, func):
		self.userMouseDrag = func

	def setKeyPress(self, func):
		self.userKeyPress = func

	def mousePress(self, pos):
		if hasattr(self, 'userMousePress'):
			self.userMousePress(pos)

	def mouseDrag(self, pos1, pos2):
		if hasattr(self, 'userMouseDrag'):
			self.userMouseDrag(pos1, pos2)

	def eventFilter(self, object, event):
		if object == self:
			if hasattr(self, 'userKeyPress') and event.type() == QtCore.QEvent.KeyPress:
				self.userKeyPress(event.key())

				return True

		elif object == self.canvas and event.type() == QtCore.QEvent.Paint:
			for i in range(self.layerList.count()):
				layer = self.layerList.item(i)

				if layer.map:
					layer.map()

			return False

		return False