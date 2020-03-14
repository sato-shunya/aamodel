# -*- coding: utf-8 -*-

import sys
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import *

from neuralNetwork import *

class weightCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def setNeuralNetwork(self, nn):
        self.nn = nn

    def initUI(self):
        self.setWindowTitle('Weight')
        self.setFixedHeight(256+10)
        self.show()

    #重みの編集 
    def mousePressEvent(self, QMouseEvent):
        N = self.nn.neurons.size
        size = 256//N 
        mx = QMouseEvent.x() // size
        my = QMouseEvent.y() // size
        print(mx,my)

        rank = self.nn.sort_neuron()[1]

        r = 2
        for ix in range(2*r):
            for iy in range(2*r):
                if (ix-r)**2 + (iy-r)**2 > r**2: continue

                x = mx + (ix-r)
                y = my + (iy-r)

                if x < 0 or y < 0 or x >= N or y >= N: continue

                X = rank[x]
                Y = rank[y]
                
                if QMouseEvent.button() == Qt.LeftButton:
                    self.nn.matrix[Y][X] = 1

                if QMouseEvent.button() == Qt.RightButton:
                    self.nn.matrix[Y][X] = -1

        self.update()


    #重みの描画
    def paintEvent(self, QPaintEvent):
        weight = self.nn.sorted_matrix()
        N = self.nn.neurons.size
        size = 256/N 

        painter = QPainter(self)
        for i in range(0,N):
            for j in range(0,N):
                w = weight[i][j]

                #x: column, y: row
                x = size*j 
                y = size*i
               
                if w > 0:
                    color = QColor(0, w*225, 0)
                else:
                    color = QColor(0, 0, -w*225)

                brush = QBrush(color, Qt.SolidPattern)
                painter.setBrush(brush)

                painter.drawRect(x,y,size,size)
