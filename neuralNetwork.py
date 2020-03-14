import json
import sys
import datetime
import os
os.environ["OMP_NUM_THREADS"] = "2"

import numpy as np
import scipy as sparse
from multiprocessing import Process

class aokiNetwork():
    #パラメータの設定だけ
    def __init__(self, setting_file):
        with open(setting_file) as js:
            self.data = json.load(js)

    #ニューロンとネットワークの設定
    def reset(self, name, param):
        #パラメータの変更
        if param != None:
            for key in param:
                if key not in self.data.keys():
                    print('error in param')
                    sys.exit()
                self.data[key] = param[key]

        self.makeNetwork()

    #ネットワークの作成
    def makeNetwork(self):
        N = self.data['N']

        #ニューロンの生成
        #一様分布 
        if self.data['phase_type'] == 'uniform':
            self.neurons = 2 * np.pi * np.random.rand(N)

        #ガウス分布
        elif self.data['phase_type'] == 'gauss':
            self.neurons = np.clip(np.random.normal(np.pi, 0.6, N), 0, 2*np.pi)

        #クラスター状態
        elif self.data['phase_type'] == 'cluster':
            nc = self.data['cluster']
            self.neurons = np.sort(np.arange(N)%nc) * (2*np.pi/nc)
        
        g_min = self.data['g_min']
        g_max = self.data['g_max']

        #重み行列の作成
        #一様分布 
        if self.data['weight_type'] == 'uniform':
            self.matrix = (g_max - g_min) * np.random.rand(N,N) + g_min

        #ガウス分布
        elif self.data['weight_type'] == 'gauss':
            self.matrix = np.clip(np.random.normal(self.data['g_ave'], self.data['g_sd'], (N,N)), g_min, g_max)

        #クラスター状態
        elif self.data['weight_type'] == 'cluster':
            nc = self.data['cluster']
            tempList = np.sort(np.arange(N)%nc)
            mat = tempList.reshape(N,1) - tempList.reshape(1,N)
            a = -np.sin((2*np.pi/nc)*mat + self.data['b'])
            self.matrix = np.where(a > 0, 1, -1)

        #対角成分をすべて0に
        np.fill_diagonal(self.matrix, 0)

    #スパイクの記録をする
    def rec_spikes(self, next_neurons):
        N = self.data['N']
        t = self.data['t']
        data = next_neurons // (2*np.pi)

        #まったく発火してない時は記録しない
        if np.all(data < 1):
            return

        #for i in range(self.data['N']):
        #    if data[i] > 0.5:
        #        self.spike_file.write(str(i) + ' ' + str(t) + '\n')

    def calc(self):
        plasticity_on = self.data['plasticity_on']

        w = self.data['w']
        dt = self.data['dt']
        N = self.neurons.size
        ep = self.data['ep']

        eia = np.exp(1.j*self.data['a'])
        eib = np.exp(1.j*self.data['b'])
        g_min = self.data['g_min']
        g_max = self.data['g_max']

        neurons = self.neurons
        matrix = self.matrix

        if plasticity_on:
            ph = np.exp(1.j*neurons)
            Cph = ph.conjugate()
            P1 = dt * (w - ((eia * ph) * np.einsum('ij,j->i', matrix, Cph)).imag / N)
            K1 = dt * (-ep * (eib * np.einsum('i,j', ph, Cph)).imag)

            ph = np.exp(1.j*(neurons + 0.5*P1))
            Cph = ph.conjugate()
            P2 = dt*(w - ((eia * ph) * np.einsum('ij,j->i', matrix + 0.5*K1, Cph)).imag / N)
            K2 = dt * (-ep * (eib * np.einsum('i,j', ph, Cph)).imag)

            ph = np.exp(1.j*(neurons + 0.5*P2))
            Cph = ph.conjugate()
            P3 = dt*(w - ((eia * ph) * np.einsum('ij,j->i', matrix + 0.5*K2, Cph)).imag / N)
            K3 = dt * (-ep * (eib * np.einsum('i,j', ph, Cph)).imag)

            ph = np.exp(1.j*(neurons + P3))
            Cph = ph.conjugate()
            P4 = dt*(w - ((eia * ph) * np.einsum('ij,j->i', matrix + K3, Cph)).imag / N)
            K4 = dt * (-ep * (eib * np.einsum('i,j', ph, Cph)).imag)

            next_neurons = (neurons + P1/6.0 + P2/3.0 + P3/3.0 + P4/6.0) % (2*np.pi)
            next_matrix = np.clip((matrix + K1/6.0 + K2/3.0 + K3/3.0 + K4/6.0), g_min, g_max)
            np.fill_diagonal(next_matrix, 0)

        else:
            ph = np.exp(1.j*neurons)
            Cph = ph.conjugate()
            P1 = dt * (w - ((eia * ph) * np.einsum('ij,j->i', matrix, Cph)).imag / N)

            ph = np.exp(1.j*(neurons + 0.5*P1))
            Cph = ph.conjugate()
            P2 = dt*(w - ((eia * ph) * np.einsum('ij,j->i', matrix, Cph)).imag / N)

            ph = np.exp(1.j*(neurons + 0.5*P2))
            Cph = ph.conjugate()
            P3 = dt*(w - ((eia * ph) * np.einsum('ij,j->i', matrix, Cph)).imag / N)

            ph = np.exp(1.j*(neurons + P3))
            Cph = ph.conjugate()
            P4 = dt*(w - ((eia * ph) * np.einsum('ij,j->i', matrix, Cph)).imag / N)

            next_neurons = (neurons + P1/6.0 + P2/3.0 + P3/3.0 + P4/6.0) % (2*np.pi)
            next_matrix = matrix

        return (next_neurons, next_matrix)

    #main loop
    def run(self):
        cnt = 0
        cnt_max = self.data['cnt_max']

        while cnt < cnt_max:
            #ニューロンの計算

            data = self.calc()
            self.neurons = data[0]
            self.matrix = data[1]
            cnt += 1

    #秩序変数を取得 
    def orderp(self, m):
        return np.abs(np.mean(np.exp(m * 1j * self.neurons)))

    #平均位相を取得 
    def mean_field(self):
        z = np.mean(np.exp(1j * self.neurons))
        return (np.abs(z) , np.angle(-z)+np.pi)

    #重みの統計情報を取得
    def stat_g(self):
        #(平均, 標準偏差, 2ループ, 3ループ)
        return (np.mean(self.matrix), \
                np.std(self.matrix), \
                np.abs(np.trace(self.matrix ** 2)) / self.data['N'])

        
    #重み分布の取得
    def get_dist(self):
        flat = self.matrix.flatten()
        return flat[np.argsort(flat)]

    #位相分布の取得
    def get_dist_phase(self):
        sort = self.sort_neuron()
        rel = sort[0]
        rank = sort[1]
        return rel[rank]

    #ニューロンをソートする関数 
    def sort_neuron(self):
        rel_neurons = (self.neurons - self.neurons[0])%(2*np.pi)
        
        #(相対位相のリスト, ランキングのリスト)
        return (rel_neurons, np.argsort(rel_neurons))

    #位相でソートされた重み行列を取得
    def sorted_matrix(self):
        sort = self.sort_neuron()
        rank = sort[1]
        return self.matrix[rank][:,rank]

    #クラスターのニューロン数を取得する
    def count_cluster(self):
        ph = self.get_dist_phase()
        n = self.neurons.size

        div = (2*np.pi / n) * 0.2
        data = np.where(ph > 2*np.pi-div, 0, ph)
        data = np.nonzero(np.where(np.diff(data) < div, 0, 1))[0]
        data = np.hstack(([-1],  data, [n-1]))
        data = np.diff(data)
    
        return data

    #最終状態での秩序変数を1次から100次まで求める
    def all_ords(self):
        ord_array = []
        for m in range(1, 101):
            ord_array.append(self.orderp(m))

        return ord_array
