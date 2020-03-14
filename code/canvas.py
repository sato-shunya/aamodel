# -*- coding: utf-8 -*-

import sys
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import *

from neuralNetwork import *

class canvas(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def setNeuralNetwork(self, nn):
        self.nn = nn

    def initUI(self):
        self.show()

    #位相の描画
    def paintEvent(self, QPaintEvent):
        phase = self.nn.get_dist_phase()

        painter = QPainter(self)
        painter.setPen(QPen(Qt.black,2,Qt.SolidLine))
        painter.drawEllipse(50,50,200,200)

        brush = QBrush(Qt.green, Qt.SolidPattern)
        painter.setBrush(brush)
        for ph in phase:
            r = 100
            x = 148 + r*np.cos(ph)
            y = 148 - r*np.sin(ph)
            painter.drawEllipse(x,y,8,8)
