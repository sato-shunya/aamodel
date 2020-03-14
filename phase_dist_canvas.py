# -*- coding: utf-8 -*-

import sys
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import *

from neuralNetwork import *

class phaseDistCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def setNeuralNetwork(self, nn):
        self.nn = nn

    def initUI(self):
        self.setWindowTitle('Phase')
        self.setFixedHeight(140)
        self.show()

    #位相の描画
    def paintEvent(self, QPaintEvent):
        phase = self.nn.get_dist_phase()/(2*np.pi)
        N = phase.size
        nc = self.nn.data['cluster']
        size = 128/N 
        h = 100

        painter = QPainter(self)

        cluster_pop= self.nn.count_cluster()
        nc = len(cluster_pop)
        painter.drawText(0, 120, str(nc))
        if nc <= 16:
            painter.drawText(20, 120, str(cluster_pop))
            painter.setPen(QPen(Qt.gray, Qt.SolidPattern))
            for i in range(nc):
                y = 100 - (h/nc)*i
                painter.drawLine(0,y,256,y)

        painter.setBrush(QBrush(Qt.blue, Qt.SolidPattern))
        painter.setPen(QPen(Qt.blue, Qt.SolidPattern))
        for i in range(0, N):
            x = 2*size*i
            y = 100 - h*phase[i] - size/2
            painter.drawEllipse(x,y,size,size)
