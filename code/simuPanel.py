import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtCore
from phase_canvas import *
from weight_canvas import *

class simuPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Aoki-Aoyagi model')

        #シミュレーション部分の作成
        self.grid = QGridLayout()
        self.setLayout(self.grid)
        self.count = 0
        self.show()

    def addSimuCanvas(self, canvas):
        cnt = self.count
        self.grid.addWidget(canvas,cnt//2, cnt%2)
        self.count += 1

    #メインループ 
    def calc(self):
        self.weightCanvas.update()
        self.phaseCanvas.update()
        self.nn.run()
