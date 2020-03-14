# -*- coding: utf-8 -*-

import sys
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import *

from neuralNetwork import *

class phaseCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def setNeuralNetwork(self, nn):
        self.nn = nn

    def initUI(self):
        self.setGeometry(400,400,5*128,5*128)
        self.setWindowTitle('Phase')
        self.show()

    #位相の描画
    def paintEvent(self, QPaintEvent):
        phase = self.nn.get_dist_phase()

        N = phase.size
        painter = QPainter(self)
        sin = -np.sin(phase.reshape(N,1) - phase.reshape(1,N))
        #sinb = -np.sin(phase.reshape(N,1) - phase.reshape(1,N) + self.nn.data['b'])
        sina = -np.sin(phase.reshape(N,1) - phase.reshape(1,N) + self.nn.data['a'])
        size = 256/N 

        if False:
            for i in range(0,N):
                for j in range(0,N):
                    w = sin[i][j]
                    x = size*j 
                    y = size*i
                   
                    if w > 0:
                        color = QColor(0, w*225, 0)
                    else:
                        color = QColor(0, 0, -w*225)

                    brush = QBrush(color, Qt.SolidPattern)
                    painter.setBrush(brush)

                    painter.drawRect(x,y,size,size)

                    if i < N-1:
                        if sina[i][j] > 0:
                            brush = QBrush(QColor(255,255,0,100), Qt.SolidPattern)
                            painter.setBrush(brush)
                            painter.drawRect(x,y,size,size)
        else:
            r = 100
            R = 120
            b = self.nn.data['b']
            a = self.nn.data['a']

            painter.setPen(QPen(Qt.red,1,Qt.SolidLine))
            painter.drawLine(148-R*np.cos(b), 148-R*np.sin(b), 148+R*np.cos(b), 148+R*np.sin(b))
            #重みの増加領域を指し示す
            painter.setPen(QPen(Qt.red,4,Qt.SolidLine))
            painter.drawLine(148, 148, 148-10*np.sin(b), 148+10*np.cos(b))

            painter.setPen(QPen(Qt.blue,1,Qt.SolidLine))
            painter.drawLine(148-R*np.cos(a), 148-R*np.sin(a), 148+R*np.cos(a), 148+R*np.sin(a))
            #位相の吸引領域を指し示す   
            painter.setPen(QPen(Qt.blue,4,Qt.SolidLine))
            painter.drawLine(148, 148, 148+10*np.sin(a), 148-10*np.cos(a))

            painter.setPen(QPen(Qt.black,2,Qt.SolidLine))
            painter.drawEllipse(48,48,2*r,2*r)
            painter.drawLine(148+r-20, 148, 148+r+20, 148)

            brush = QBrush(Qt.green, Qt.SolidPattern)
            painter.setBrush(brush)
            for ph in phase:
                x = 148 + r*np.cos(ph)-4
                y = 148 - r*np.sin(ph)-4
                painter.drawEllipse(x,y,8,8)
