import json
import sys
import datetime
import os

import numpy as np
import scipy as sparse
from multiprocessing import Process
import neuralNetwork

class analyse:
    def __init__(self, datadir='notitle'):
        with open('analyse_setting.json') as js:
            self.multi_data = json.load(js)
            if datadir == 'notitle':
                self.datadir = str(datetime.datetime.today().strftime("%Y_%m_%d_%H%M"))
            else:
                self.datadir = datadir

    #サンプルデータの作成 
    def makeSample(self, nn, dirname, cnt_max):
        #データ保存処理
        def makeDataFiles(nn, cnt):
            np.savetxt(dirname+'/distphase'+str(cnt), nn.get_dist_phase())
            np.savetxt(dirname+'/distmat'+str(cnt), nn.get_dist())
            #np.savetxt(dirname+'/sorted_matrix'+str(cnt), nn.sorted_matrix())
            np.savetxt(dirname+'/ord'+str(cnt), nn.all_ords())
            np.savetxt(dirname+'/cluster_data'+str(cnt), nn.count_cluster())

        #定点観測
        for cnt in range(0, cnt_max):
            makeDataFiles(nn, cnt)
            nn.run()

        makeDataFiles(nn, cnt_max)

        #パラメータ情報も記録 
        with open(dirname+'/setting.json', 'w') as jsonfile:
            json.dump(nn.data, jsonfile, indent=2, ensure_ascii=False)

    #パラメータを設定したニューラルネットワークを生成
	#パラメータは円周率がかかっている
    def makeNN(self, a, b):
        nn = neuralNetwork.aokiNetwork('setting.json')
        nn.data['a'] = a
        nn.data['b'] = b
        nn.makeNetwork()

        return nn

    #いろんなパラメータを走査する
    def scanAllParam(self):
        DeltaA = self.multi_data['a_max'] - self.multi_data['a_min']
        DeltaB = self.multi_data['b_max'] - self.multi_data['b_min']
        a_min = self.multi_data['a_min']
        b_min = self.multi_data['b_min']

        split_a = self.multi_data['split_a']
        split_b = self.multi_data['split_b']

        #走査処理
        os.mkdir(self.datadir)
        for ia in range(split_a+1):
            for ib in range(split_b+1):
                a = a_min + (0 if split_a == 0 else (ia/split_a))*DeltaA
                b = b_min + (0 if split_b == 0 else (ib/split_b))*DeltaB
                dirname = self.datadir + '/a' + str(ia) + '_b' + str(ib)

                self.multiProcess(dirname, a, b)

    #一つのパラメータについて分析 
	#パラメータは円周率がかかっている
    def scanParam(self, a, b):
        os.mkdir(self.datadir)
        dirname = self.datadir + '/a' + str(a) + '_b' + str(b)
        self.multiProcess(dirname, a, b)

    #並列処理でサンプル作成 
	#パラメータは円周率がかかっている
    def multiProcess(self, parent_dir, a, b):
        process_list = []
        sample = self.multi_data['sample']
        cnt_max = self.multi_data['cnt_max']
        core = self.multi_data['core']
        core_cnt = 0

        os.mkdir(parent_dir)
        for i in range(sample):
            #サンプル用のディレクトリ作成
            dirname= parent_dir + '/sample' + str(i)
            os.mkdir(dirname)

            #サンプルNNの作成
            nn = self.makeNN(a, b)

            #コアプロセスの設定
            p = Process(target=self.makeSample, args=([nn, dirname, cnt_max]))
            p.start()
            process_list.append(p)
            core_cnt += 1

            #サブプロセスが全て完了するまで待機
            if core_cnt == core:
                core_cnt = 0
                for i in range(len(process_list)):
                    process_list[i].join()
                process_list = []

if __name__ == '__main__':
    args = sys.argv

    #ディレクトリ名だけ指定 scanAllParam
    if len(args) == 2:
        a = analyse(args[1])
        a.scanAllParam()

    #パラメータを直接指定
    elif len(args) == 3:
        a = analyse()
        a.scanParam(np.pi*float(args[1]), np.pi*float(args[2]))

    #ディレクトリ名を指定 かつ パラメータを直接指定
    elif len(args) == 4:
        a = analyse(args[1])
        a.scanParam(np.pi*float(args[2]), np.pi*float(args[3]))
    
    #scanAllParam
    else:
        a = analyse()
        a.scanAllParam()
