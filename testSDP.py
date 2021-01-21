'''
求解问题部分SDP
软件缺陷预测数学描述
'''

import numpy as np
import MOEADutl
import methods

# 函数的维度（目标维度不一致的自行编写目标函数）
Dimention = 10
# 函数目标个数
Func_num = 2
# 函数的变量的边界,GetDataSetSize(XX)函数
Bound = [0, 1]
# Bound = [start_bound, end_bound]
gx = -1


# global X  # 初始特征，全部的特征矩阵
#
###########   自定义
# method = "ensemble_heter"
# n_clfs = 3
# fs_functions = "fisher"
# score_name = "auc"


def Func(seed, XX, y, method, n_clfs, fs_functions, score_name):  # XX:原始特征数据集,seed为变量(种群个体)
    # 2目标函数

    # 全0的需要去掉
    if MOEADutl.isSeed(seed) != True:
        return [GetDataSetSize(XX), 0]
    # end_bound = GetDataSetSize(XX)
    # global XX = XXX
    global X, cnt  # 对应seed的特征子集，全局变量，为了F1(X)可以有效
    X, cnt = MOEADutl.seedtovector(seed, XX)  # 根据种群个体获得的特征子集
    f1 = F1(seed)  # 优化目标1，特征数量
    # f1 = len(X)
    if f1 <= 2:  # 特征数少于2个不做评价,直接最差点处理
        return [GetDataSetSize(XX), 0]

    scores = methods.run_method(method, X, y, n_clfs=n_clfs, fs_functions=fs_functions, score_name=score_name)
    f2 = F2(scores)  # 优化目标2，模型AUC值
    print("[", f1, ",", f2, "]")
    return [f1, f2]


def F1(seed):  # 返回特征子集规模，即特征数量
    # X = MOEADutl.seedtovector(seed, XX)  # 根据种群个体获得的特征子集
    f = cnt
    return f


def F2(scores):  # 返回特征子集对应的模型AUC值、
    f = np.max(scores)
    return f


def GetInfo(seed, XX, y, method, n_clfs, fs_functions, score_name):
    return seed, XX, y, method, n_clfs, fs_functions, score_name


def GetDataSetSize(XX):  # 返回原始数据集的规模，用于生成初始个体
    return len(XX[0])
