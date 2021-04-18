import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import re
import utilities


# 数据线处理一下，去重，转float类型，datasetName数据集名称
def getDataset_HUatu(datasetName):
    pattern = re.compile(r'\[\s*(\d+)\s*\,\s*(\d+.\d+)\s*\]')
    content = ""  # 存放文件内容
    # 文件所在目录
    path = "D:/PycharmProjects/software_defect_prediction-master/logfile/"
    # datasetName = 'PC4_RF'  # 数据集名称，对应txt文件名
    with open(path + datasetName + '.txt', 'r') as f:
        content = f.read()

    str = pattern.findall(content)
    temp = np.array(str)
    generate_Seed_Num = temp.shape  # 产生候选解个数的数目
    temp = np.array(list(set([tuple(t) for t in temp])))  # 去重
    temp = temp.astype(float)

    # temptemp矩阵只保留最大值
    temptemp = temp
    # temptemp = temptemp[:, temptemp[0].argsort()]  # 按第一行排序
    temptemp = temptemp[temptemp[:, 0].argsort()]  # 按照第1列对行排序
    print(temptemp)
    lenSize = len(temptemp[:, 0])  # 边界
    del_index = []  # 记录要删除的行号
    ans = []
    index = 0
    # jindex = index
    if lenSize >= 2:
        # for index in range(lenSize - 1):
        while index <= (lenSize - 1):  # 下标为0到倒数第二个，为了第二个while循环中jindex+1遍历比较
            submax = temptemp[index][1]
            tempindex = index
            jindex = index
            while jindex + 1 < lenSize and temptemp[jindex][0] == temptemp[jindex + 1][0]:
                if submax >= temptemp[jindex + 1][1]:
                    del_index.append(jindex + 1)
                else:
                    submax = temptemp[jindex + 1][1]
                    tempindex = jindex + 1
                    del_index.append(jindex)
                jindex = jindex + 1
            print(tempindex, "   ", temptemp[tempindex], "    ", submax)
            ans.append(temptemp[tempindex])

            index = jindex + 1

    print(del_index)
    ans = np.array(ans)
    print(ans)
    # for index in range(len(temptemp[:, 0])):
    temptemp = ans  # 矩阵最大坐标集合

    print("temptemp", temptemp)
    #
    print("解集矩阵：", temp)
    # 查看解的个数
    print("产生候选解个数的变化temptemp", generate_Seed_Num, "temp", temptemp.shape)
    print("数据集", datasetName)
    # 第一列,用去重后的
    number = temptemp[:, 0]
    print(number)
    # 第二列
    auc = temptemp[:, 1]
    print(auc)
    print("maxAUC:", max(auc))
    # print(pattern.findall(content))
    print(type(str))
    # 返回三个矩阵
    return number, auc, temptemp  # temptemp


# 数据先处理一下，找解的集合数组
def getJieJI_Huatu(datasetName):
    pattern = re.compile(r':\s*([0-1][0-1][0-1][0-1][0-1][0-1][0-1][0-1][0-1]+)')
    content = ""  # 存放文件内容
    # 文件所在目录
    path = "D:/PycharmProjects/software_defect_prediction-master/logfile/"
    # datasetName = 'PC4_RF'  # 数据集名称，对应txt文件名
    with open(path + datasetName + '.txt', 'r') as f:
        content = f.read()

    str = pattern.findall(content)
    temp = np.array(str)
    # temp = temp.astype(float)
    # temp = np.array(list(set([tuple(t) for t in temp])))  # 去重
    temp = np.unique(temp)
    # temp = temp.astype(float)
    # print(temp)
    length = len(temp[0])  # 数组长度
    recoder = [0] * length  # 记录数组
    for item in temp:
        for i in range(length):
            if item[i] == '1':
                recoder[i] = recoder[i] + 1
            else:
                continue

    print(recoder)  # 输出记录数组
    # 1000100000010001000100000000000000000
    # 0000000000010101000100000000000000000
    # 0000000100010000001000100000001000100
    # 11001110010100011111111111010001010100
    # 00000000000001000000000000010010000110
    # 00000000000000000100100000000001001000
    # 01100000000000001001000000000000010010

    # print(temp)
    # 查看解的个数
    print(temp.shape)
    # 第一列
    # number = temp[:, 0]
    # print(number)
    # 第二列
    # auc = temp[:, 1]
    # print(auc)
    # print(pattern.findall(content))
    # print(type(str))
    # 返回三个矩阵
    return temp, recoder


# 数据先处理一下，找解的集合交集和合集
def getJieJI_Jiaoji_Heji_Huatu(datasetName):
    res, recoder = getJieJI_Huatu(datasetName)
    print(res)

    recoder = np.array(recoder)
    recoder = recoder.astype(str)

    res1 = ''.join(recoder)
    resjiaoji, resheji = res1, res1
    print("解集为", res1)

    # 交集
    stringjiaoji = list(resjiaoji)
    for index in range(len(resjiaoji)):
        if stringjiaoji[index] == '1':
            stringjiaoji[index] = '0'
        elif stringjiaoji[index] == '0':
            continue
        else:
            stringjiaoji[index] = '1'

    resjiaoji = ''.join(stringjiaoji)
    print("交集为", resjiaoji)

    # 合集
    stringheji = list(resheji)
    for index in range(len(resheji)):
        if stringheji[index] == '0':
            continue
        else:
            stringheji[index] = '1'

    resheji = ''.join(stringheji)
    print("合集为", resheji)

    return res, res1, resjiaoji, resheji


# 从解集中获取频率高的特征
def getFeatures(datasetName, datasetOriginalName):  # datasetOriginalName为匹配软件度量元名字用的数据集
    res, recoder = getJieJI_Huatu(datasetName)
    path = "D:/PycharmProjects/software_defect_prediction-master/datasets/"
    # df = pd.read_csv(path + "MC1.arff.csv")   # NASA MDP数据集
    df = pd.read_csv(path + datasetOriginalName + ".csv")  # PROMISE数据集
    labels = list(df.columns.values)
    labels.pop()  # 删除最后一个元素“Defective”
    print(labels)
    print("________________________________________")
    recoder = np.array(recoder).reshape(1, -1)
    labels = np.array(labels).reshape(1, -1)
    print(recoder.shape, recoder)
    print(labels.shape, labels)

    # np.hstack(recoder, labels)
    data = np.row_stack((recoder, labels))
    print(data, data.shape)
    data = data[:, data[0].argsort()[::-1]]  # 按第一行排序，[::-1]表示降序
    print(data)
    return data, recoder, labels
    #
    # results = np.array(r)
    # print(results)


# 四种方法合并后排序
def getAllFeatures(datasetOriginalName):
    # datasetOriginalName = "ant"
    methods1 = "KNN"
    methods2 = "Pearson"
    methods3 = "Greedy"
    methods4 = "EL"
    data1, r1, l1 = getFeatures(datasetOriginalName + "_" + methods1, datasetOriginalName)
    data2, r2, l2 = getFeatures(datasetOriginalName + "_" + methods2, datasetOriginalName)
    # data3,r3,l3 = getFeatures(datasetOriginalName + "_" + methods3, datasetOriginalName)
    data4, r4, l4 = getFeatures(datasetOriginalName + "_" + methods4, datasetOriginalName)
    #
    print("+++++++++四个方法的统计结果累加+++++++++++++++")
    data = data1  # 初始化
    data = np.append(data, data2, axis=1)
    # data = np.append(data, data3, axis=1)
    data = np.append(data, data4, axis=1)
    #
    print(r2, type(r2), "++++++")
    recoder = r1
    recoder = np.append(recoder, r2, axis=0)
    # recoder = np.append(recoder,r3,axis=0)
    recoder = np.append(recoder, r4, axis=0)
    print(recoder, type(recoder))
    print(recoder.sum(axis=0))
    recoder = recoder.sum(axis=0)
    data = np.row_stack((recoder, l1))

    # data = pd.DataFrame(data)
    a = data[0].astype(np.int)
    print(a.dtype)
    print(data, data.shape)
    # data = data[:, data[0].argsort()[::-1]]  # 按第一行排序，[::-1]表示降序
    data = data[:, a.argsort()[::-1]]  # 按第一行排序，[::-1]表示降序
    print(data)
    # print("+++++++++四个方法的统计结果累加+++++++++++++++")
    # allData = data1  # 初始化
    # # print(np.concatenate((data, data2), axis=0))
    # allData = np.append(allData, data2, axis=1)
    # # allData = np.append(allData, data3, axis=1)
    # allData = np.append(allData, data4, axis=1)
    # print(allData)


    return data




if __name__ == '__main__':
    # test测试
    # print(getDataset_HUatu("MC1_W_SVM"))
    # r, r1, r2 = getJieJI_Jiaoji_Heji_Huatu("PC2_Greedy")
    # print(r, r1, r2)
    # r1 = "011010000100100000000001001010011100"
    # r2 = "011010111111110010101001011011011100"
    # 一、获取软件度量元

    data = getAllFeatures("ant")
    print(data, type(data), data.shape, type(data[1][1]), type(data[0][1]))
    for i in range(len(data[1])):
        print(data[1][i], "\t", data[0][i])

    # print(data[1:])
    # path = "D:/PycharmProjects/software_defect_prediction-master/logfile/"
    # np.savetxt(path + 'data.out', data, delimiter=',')  # X is an array

    # res, recoder = getJieJI_Huatu("PC5_RF")
    # print(res,recoder)
