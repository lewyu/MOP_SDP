import testSDP
import utils
import utils.GA_Utils as GA_Utils
from utils import Utils
import time
import sys
import argparse
import utilities as ut
import testSDP

import numpy as np
import pylab as pl

import methods
import numpy as np
import pylab as pl

import SDPlog

from sklearn.model_selection import cross_val_score
from scipy.stats.stats import pearsonr
from sklearn.metrics import mutual_info_score
import pandas as pd
from scipy.sparse import issparse
import ml_metrics  # 度量指标
from sklearn import metrics
from scipy.io import arff
from sklearn.metrics import recall_score
from sklearn.metrics import fbeta_score, make_scorer
import time
import MOEADutl
import ReadDataSets
from pybloom_live import ScalableBloomFilter

# 控制台打印输出
path = "D:/PycharmProjects/software_defect_prediction-master/logfile/"
sys.stdout = SDPlog.Logger(path + 'camelOSA_EL.txt', sys.stdout)
sys.stderr = SDPlog.Logger(path + 'a.log_err_file.txt', sys.stderr)


class MOEAD:
    # 数据集读取
    XX = ReadDataSets
    # 布隆过滤器
    # sbf = ScalableBloomFilter(mode=ScalableBloomFilter.SMALL_SET_GROWTH)

    # 0表示最小化目标求解，1最大化目标求解。（约定），和理想点设置有关
    # problem_type = 0
    # problem_type=1
    # 测试函数
    Test_fun = testSDP
    # 动态展示的时候的title名称,优化问题的名字，同时也是“vector_csv_file”中的名字
    name = 'SDP'
    # 使用那种方式、DE/GA 作为进化算法
    # GA_DE_Utils = Utils.DE_Utils
    GA_DE_Utils = Utils.GA_Utils

    # 种群大小，取决于vector_csv_file/下的xx.csv
    Pop_size = -1
    # 最大迭代次数
    max_gen = 20
    # 邻居设定（只会对邻居内的相互更新、交叉）
    T_size = 5
    # 支配前沿ID
    EP_X_ID = []
    # 支配前沿 的 函数值
    EP_X_FV = []

    # 种群
    Pop = []
    # Pop = list(set(Pop))
    Pop = set(Pop)
    # 种群计算出的函数值
    Pop_FV = []
    # 权重
    W = []
    # 权重的T个邻居。比如：T=2，(0.1,0.9)的邻居：(0,1)、(0.2,0.8)。永远固定不变
    W_Bi_T = []
    # 理想点。（比如最小化，理想点是趋于0）
    # ps:实验结论：如果你知道你的目标，比如是极小化了，且理想极小值(假设2目标)是[0,0]，
    # 那你就一开始的时候就写死moead.Z=[0,0]吧
    Z = [0, 1]
    # 权重向量存储目录
    csv_file_path = 'D:/PycharmProjects/software_defect_prediction-master/vector_csv_file'
    # 当前迭代代数
    gen = 0
    # 是否动态展示
    # need_dynamic = False
    need_dynamic = True
    # 是否画出权重图
    draw_w = True
    # 用于绘图：当前进化种群中，哪个，被，正在 进化。draw_w=true的时候可见
    now_y = []

    # draw_w=True

    def __init__(self):
        self.Init_data()

    def Init_data(self):
        # 加载数据集并打印输出
        print("数据集：", self.XX.dataset_name)

        # 加载权重
        Utils.Load_W(self)
        # 计算每个权重Wi的T个邻居
        Utils.cpt_W_Bi_T(self)
        # 创建种群
        self.GA_DE_Utils.Creat_Pop(self)
        # 初始化Z集，最小问题0,0
        Utils.cpt_Z(self)

    def show(self):
        if self.draw_w:
            Utils.draw_W(self)
        Utils.draw_MOEAD_Pareto(self, moead.name + "num:" + str(self.max_gen) + "")
        Utils.show()

    def run(self):
        t = time.time()
        # EP_X_ID：支配前沿个体解，的ID。在上面数组：Pop，中的序号
        # envolution开始进化
        EP_X_ID = self.GA_DE_Utils.envolution(self)
        print('你拿以下序号到上面数组：Pop中找到对应个体，就是多目标优化的函数的解集啦!')
        print("支配前沿个体解，的ID（在上面数组：Pop，中的序号）：", EP_X_ID)
        # 打印输出
        # cnt = 0
        for ID in EP_X_ID:
            # if (moead.Pop_FV[ID] in moead.sbf):
            #     continue
            # else:
            #     moead.sbf.add(moead.Pop_FV[ID])
            #     cnt = cnt + 1
            print("POPSeed:", moead.Pop[ID], "FV结果：", moead.Pop_FV[ID])

        # print("去重后最优解个数为：", cnt)
        dt = time.time() - t
        print("用时：", dt, "秒")
        self.show()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-fs', '--fs_functions', nargs="+", required=True,
    #                     choices=["pearson", "fisher", "greedy"])
    #
    # args = parser.parse_args()
    # fs_functions = args.fs_functions
    # ##### 2. ------- RUN TRANING METHOD
    # method = "forward_selection"
    # # method = "ensemble_heter"
    # n_clfs = 3
    # # fs_functions = "fisher"
    # score_name = "auc"

    start = time.time()
    # dataset_name = "PC2"
    # ###########
    moead = MOEAD()
    # @#￥#######

    # XX, y, ft_names = ut.read_dataset("D:/PycharmProjects/software_defect_prediction-master/datasets/",
    #                                   dataset_name=dataset_name)
    # seed = "110011001100110011001100110011001100"
    # seed = "111100001111000000000000001110001100"
    # 1、获取种群个体对应的特征子集
    # X = seedtovector(seed, X)
    # print(XX)
    # print(y)
    ################## testSDP
    # seed = MOEADutl.Creat_child(moead, XX)
    #
    # print(testSDP.Func(seed, XX, y, method=method, n_clfs=n_clfs,
    #                    fs_functions=fs_functions,
    #                    score_name=score_name))
    #
    # print(testSDP.GetDataSetSize(XX))
    # print(MOEADutl.Creat_child(X))
    # pl.title(dataset_name)
    # pl.ylabel("AUC")

    # RUN method

    # scores = methods.run_method(method, X, y, n_clfs=n_clfs,
    #                             fs_functions=fs_functions,
    #                             score_name=score_name)
    #
    # print(max(scores))  # 后加的,调试用的
    pl.legend(loc="best")
    pl.show()

    print('耗时：', time.time() - start, '秒')

    # np.random.seed(1)
    # moead = MOEAD()
    moead.run()
