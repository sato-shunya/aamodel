# -*- coding: utf-8 -*-

import sys
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import *

from neuralNetwork import *

class weightDistCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def setNeuralNetwork(self, nn):
        self.nn = nn

    def initUI(self):
        self.setWindowTitle('Phase')
        self.setFixedHeight(110)
        self.show()

    #位相の描画
    def paintEvent(self, QPaintEvent):
        weight = (self.nn.get_dist()+1)/2
        N = weight.size
        size = 256/self.nn.neurons.size
        h = 100

        painter = QPainter(self)
        painter.setBrush(QBrush(Qt.red, Qt.SolidPattern))
        painter.setPen(QPen(Qt.red, Qt.SolidPattern))
        for i in range(0, N):
            x = size*i/np.sqrt(N)
            y = 100 - h*weight[i]
            painter.drawEllipse(x,y,1,1)
