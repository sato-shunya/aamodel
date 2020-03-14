import sys
import numpy as np
import time
import neuralNetwork

def makeNN(N):
    nn = neuralNetwork.aokiNetwork('setting.json')
    nn.data['a'] = np.pi*0.1
    nn.data['b'] = -np.pi*0.2
    nn.data['N'] = N
    nn.makeNetwork()

    return nn

realtime = time.time()
for i in range(2,10):
    nn = makeNN(2**i)
    nn.run()
    print(i, time.time() - realtime)
