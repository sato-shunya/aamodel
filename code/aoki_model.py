import sys
import numpy as np
import time
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtCore

from phase_canvas import *
from weight_canvas import *
from phase_dist_canvas import *
from weight_dist_canvas import *
from simuPanel import *

class mainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.timer = None
        self.makeNeuralNetwork()
        self.realtime = time.time()
        self.realcnt = 0

    def initUI(self):
        self.setGeometry(400,400,700,400)
        self.setWindowTitle('Aoki-Aoyagi model')

        self.phaseCanvas = phaseCanvas()
        self.weightCanvas = weightCanvas()
        self.phaseDistCanvas = phaseDistCanvas()
        self.weightDistCanvas = weightDistCanvas()

        #ツールバーの作成
        toolbar = QToolBar()

        self.startButton = QPushButton('start')
        self.stopButton = QPushButton('stop')
        self.setButton = QPushButton('set')
        self.alpha_input = QLineEdit('0.1')
        self.alpha_input.setFixedWidth(50)
        self.beta_input = QLineEdit('-0.2')
        self.nNum_input = QLineEdit('64')
        self.beta_input.setFixedWidth(50)
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.beta_slider = QSlider(Qt.Horizontal)
        self.nNum_input.setFixedWidth(50)
        self.plastic_check = QCheckBox('plas.')
        self.plastic_check.setChecked(True)
        
        #
        self.setSliderValue()

        #ComboBox
        self.clusterComboBox = QComboBox()
        self.clusterComboBox.addItem('random')
        [self.clusterComboBox.addItem(str(i)) for i in range(2,17)]

        toolbar.addWidget(self.startButton)
        toolbar.addWidget(self.stopButton)
        toolbar.addWidget(self.setButton)
        toolbar.addWidget(QLabel('  a:'))
        toolbar.addWidget(self.alpha_input)
        toolbar.addWidget(QLabel('  b:'))
        toolbar.addWidget(self.beta_input)
        toolbar.addWidget(QLabel('  Nc:'))
        toolbar.addWidget(self.clusterComboBox)
        toolbar.addWidget(QLabel('  N:'))
        toolbar.addWidget(self.nNum_input)
        toolbar.addWidget(QLabel(' '))
        toolbar.addWidget(self.plastic_check)
        self.addToolBar(toolbar)

        #シミュレーション部分の作成
        simu_panel = simuPanel()
        simu_panel.addSimuCanvas(self.alpha_slider)
        simu_panel.addSimuCanvas(self.beta_slider)
        simu_panel.addSimuCanvas(self.phaseDistCanvas)
        simu_panel.addSimuCanvas(self.weightDistCanvas)
        simu_panel.addSimuCanvas(self.phaseCanvas)
        simu_panel.addSimuCanvas(self.weightCanvas)
        self.setCentralWidget(simu_panel)

        #ボタンのスロット接続
        self.startButton.clicked.connect(self.start)
        self.stopButton.clicked.connect(self.stop)
        self.setButton.clicked.connect(self.setButtonPressed)
        self.alpha_slider.valueChanged.connect(self.setBySlider)
        self.beta_slider.valueChanged.connect(self.setBySlider)
        self.plastic_check.stateChanged.connect(self.changePlasticity)

        self.clusterComboBox.currentIndexChanged.connect(self.setCluster)

    def makeNeuralNetwork(self):
        self.nn = aokiNetwork('setting.json')
        self.nn.reset(None,None)
        self.phaseCanvas.setNeuralNetwork(self.nn)
        self.weightCanvas.setNeuralNetwork(self.nn)
        self.phaseDistCanvas.setNeuralNetwork(self.nn)
        self.weightDistCanvas.setNeuralNetwork(self.nn)

    def setSliderValue(self):
        ap = 100 * float(self.alpha_input.text())
        bp = (float(self.beta_input.text()) + 1)*50
        self.alpha_slider.setValue(ap)
        self.beta_slider.setValue(bp)

    def changePlasticity(self):
        if self.plastic_check.isChecked():
            self.nn.data['plasticity_on'] = True
        else:
            self.nn.data['plasticity_on'] = False

    def setCluster(self):
        nc = self.clusterComboBox.currentText()

        if nc == 'random':
            self.nn.data['weight_type'] = 'uniform'
            self.nn.data['phase_type'] = 'uniform'

        else:
            self.nn.data['weight_type'] = 'cluster'
            self.nn.data['phase_type'] = 'cluster'
            self.nn.data['cluster'] = int(nc)

        if self.timer != None:
            self.stop()
        self.nn.makeNetwork()
        self.weightCanvas.update()
        self.phaseCanvas.update()
        self.phaseDistCanvas.update()
        self.weightDistCanvas.update()

    def setBySlider(self):
        a = float(self.alpha_slider.value())/100
        b = (float(self.beta_slider.value())-50.0)/50
        self.alpha_input.setText(str(a))
        self.beta_input.setText(str(b))

        self.setValue(a,b)

    def setButtonPressed(self):
        a = float(self.alpha_input.text())
        b = float(self.beta_input.text())
        self.setValue(a,b)

    def setValue(self, a, b):
        self.nn.data['a'] = a * np.pi
        self.nn.data['b'] = b * np.pi

        self.weightCanvas.update()
        self.phaseCanvas.update()

    def start(self):
        a = float(self.alpha_input.text())
        b = float(self.beta_input.text())
        self.nn.data['a'] = a * np.pi
        self.nn.data['b'] = b * np.pi
        self.setTimer()

    def stop(self):
        self.stopTimer()

    #タイマーを設定
    def setTimer(self):
        self.time = 0
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.calc)
        self.timer.start(100)

    #タイマーを停止 
    def stopTimer(self):
        self.timer.stop()

    #メインループ 
    def calc(self):
        print(time.time() - self.realtime, self.realcnt)
        self.realcnt += 1
        self.weightCanvas.update()
        self.phaseCanvas.update()
        self.phaseDistCanvas.update()
        self.weightDistCanvas.update()
        self.setStatusBarText()
        self.nn.run()

    #ステータスバーの更新
    def setStatusBarText(self):
        order = np.round([self.nn.orderp(n) for n in range(1,5)],2) 
        self.statusBar().showMessage('time:' +str(time.time() - self.realtime))

    def closeEvent(self, event):
        QCoreApplication.instance().quit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = mainWindow()
    window.show()
    sys.exit(app.exec_())
