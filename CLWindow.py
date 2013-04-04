__author__ = 'Marc de Klerk'

import sys

from PyQt4 import QtCore, QtGui

import pyopencl as cl
import numpy as np

try:
	from OpenGL.GL import *
except ImportError:
	raise ImportError("Error importing PyOpenGL")

cm = cl.mem_flags
szFloat = np.dtype(np.float32).itemsize
szInt = np.dtype(np.int32).itemsize

class CLWindow(QtGui.QMainWindow):
	class CenteredScrollArea(QtGui.QScrollArea):
		def __init__(self, parent=None):
			QtGui.QScrollArea.__init__(self, parent)

		def eventFilter(self, object, event):
			if object == self.widget() and event.type() == QtCore.QEvent.Resize:
				QtGui.QScrollArea.eventFilter(self, object, event)
			else:
				QtGui.QScrollArea.eventFilter(self, object, event)

			return True

	def __init__(self, canvas):
		super(CLWindow, self).__init__(None)

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
#		self.layerList.indexesMoved.connect(self.sigLayerIndexesMoved)

		self.canvas = canvas
		self.canvas.mousePress = self.mousePress
		self.canvas.mouseDrag = self.mouseDrag

		self.scrollarea = QtGui.QScrollArea()
		#self.scrollarea = self.CenteredScrollArea()
		self.scrollarea.setWidget(self.canvas)
		self.scrollarea.setAlignment(QtCore.Qt.AlignCenter)

		layoutButtons = QtGui.QVBoxLayout()
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

		self.setCentralWidget(splitter)

		self.installEventFilter(self)

		splitter.setSizes([100, self.size().width()-100])

	def addButton(self, name, action):
		btn = QtGui.QPushButton(name)
		self.widgetButtons.layout().addWidget(btn)

		btn.clicked.connect(action)

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

	def addLayer(self, name, clobj, shape=None, opacity=None, datatype=None, filter=None):
		layer = self.canvas.addLayer(clobj, shape, opacity, datatype, filter=filter)

		item = QtGui.QListWidgetItem(name)
		item.setText(name)
		item.setFlags(	QtCore.Qt.ItemIsUserCheckable |
						QtCore.Qt.ItemIsDragEnabled |
						QtCore.Qt.ItemIsSelectable |
						QtCore.Qt.ItemIsEnabled)
		item.setCheckState(QtCore.Qt.Checked)
		item.layer = layer

		self.layerList.addItem(item)

	def sigCurrentLayerChanged(self, item):
		layer = item.layer

	def sigLayerOpacity(self, value):
		if self.layerList.currentItem() == None:
			return

		item = self.layerList.currentItem().layer
		item.opacity = float(value)/100

		self.updateCanvas()

	def sigZoom(self, value):
		self.setZoom(value)

	def sigLayerClicked(self, item):
		checked = item.checkState() == QtCore.Qt.Unchecked
		layer = item.layer

		if layer.enabled == checked:
			layer.enabled = not layer.enabled
			self.updateCanvas()
		else:
			self.slider.setValue(100*layer.opacity)

#	def sigLayerIndexesMoved(self):
#		del self.canvas.layers[:]
#
#		for i in range(self.layerList.count()):
#			layer = self.layerList.item(i).layer
#			self.canvas.layers.append(layer)
#
#		print self.layerList
#
#		self.updateCanvas()

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
		if hasattr(self, 'userKeyPress') and event.type() == QtCore.QEvent.KeyPress:
			self.userKeyPress(event.key())

			return True

		return False

	def drawLayer(self):
		for i in range(self.layerList.count()):
			layer = self.layerList.item(i)

			if layer.map:
				layer.map()
			else:
				pass
