import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import HuatuUtil

# 绘制散点图与分布图

# 设施数据集名称

# datasetName = "CM1"
# datasetName = "KC3"
# datasetName = "MC1"
# datasetName = "MC2"
# datasetName = "MW1"
# datasetName = "PC2"
# datasetName = "PC4"
datasetName = "PC5"
# 获取数据集，通过正则表达式匹配出解集，并去后转为float型矩阵，number是第一列，auc为第二列，temp是返回的矩阵
number1, auc1, temp1 = HuatuUtil.getDataset_HUatu(datasetName + '_Pearson')
number2, auc2, temp2 = HuatuUtil.getDataset_HUatu(datasetName + '_Greedy')
number3, auc3, temp3 = HuatuUtil.getDataset_HUatu(datasetName + '_EL')  # 是EL,包含随机森林
# number4, auc4, temp4 = HuatuUtil.getDataset_HUatu(datasetName + '_RF') #naive方法是直接用的随机森林，只跑了KC3的，RF文件没有随机森林
number5, auc5, temp5 = HuatuUtil.getDataset_HUatu(datasetName + '_KNN')  # 原来是KNN，这里做成MOP_Naive
# 转成df矩阵，方便seaborn散点图可视化工具
df1 = DataFrame(temp1, columns=["Number of retained features", "AUC"])
df2 = DataFrame(temp2, columns=["Number of retained features", "AUC"])
df3 = DataFrame(temp3, columns=["Number of retained features", "AUC"])
# df4 = DataFrame(temp4, columns=["Number of retained features", "AUC"])
df5 = DataFrame(temp5, columns=["Number of retained features", "AUC"])

# 1.seaborn散点图可视化工具
# sns.pairplot(df1)
# sns.pairplot(df2)
# sns.pairplot(df3)
# sns.jointplot(x="Number of retained features", y="AUC", data=df1, kind="kde")  # 双变量分布
# sns.jointplot(x="Number of retained features", y="AUC", data=df2, kind="kde")  # 双变量分布
# sns.jointplot(x="Number of retained features", y="AUC", data=df3, kind="kde")  # 双变量分布
# sns.set_style('darkgrid')  # 设置网格

# sns.regplot(x="Number of retained features", y="AUC", data=df1)  # 简单线性拟合
# sns.regplot(x="Number of retained features", y="AUC", data=df2)  # 简单线性拟合
# sns.regplot(x="Number of retained features", y="AUC", data=df3)  # 简单线性拟合
# sns.relplot(x="Number of retained features", y="AUC", kind="line", data=df1)  # 双变量-三变量连续图
# sns.relplot(x="Number of retained features", y="AUC", kind="line", data=df2)  # 双变量-三变量连续图
# sns.relplot(x="Number of retained features", y="AUC", kind="line", data=df3)  # 双变量-三变量连续图

# 2。不用seaborn散点图可视化工具，直接matplotlib画图
fig = plt.figure()
ax1 = fig.add_subplot(111)
# 设置标题
ax1.set_title(datasetName)
# 设置X轴标签
plt.xlabel('Number of retained features')
# 设置Y轴标签
plt.ylabel('AUC')
# 画散点图，s设置点的大小 c是颜色 alpha是透明度
ax1.scatter(number1, auc1, c='r', s=100, marker='.', label=datasetName + '_MOP_PRF', alpha=0.5)
ax1.scatter(number2, auc2, c='b', s=100, marker='*', label=datasetName + '_MOP_FSG', alpha=0.5)
ax1.scatter(number3, auc3, c='g', s=100, marker='+', label=datasetName + '_MOP_EL', alpha=0.5)  # 集成学习，之前跑实验写错名字了
# ax1.scatter(number4, auc4, c='y', s=10, marker='o', label=datasetName + '_RF', alpha=0.5)  # 没有随机森林的EL
ax1.scatter(number5, auc5, c='#000000', s=100, marker='v', label=datasetName + '_MOP_Naive', alpha=0.5)
# 折线图
plt.plot(number1, auc1, c='r')
plt.plot(number2, auc2, c='b')
plt.plot(number3, auc3, c='g')
# plt.plot(number4, auc4, c='y')
plt.plot(number5, auc5, c='#000000')
# plt.plot(x,y)
# 设置图标
plt.legend()
# 显示所画的图
plt.show()

if __name__ == '__main__':
    print("hello")
