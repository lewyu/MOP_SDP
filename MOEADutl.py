import numpy as np
import math
import testSDP


#### MOEA/D
def isSeed(seed):  # 判断是否为有效种子,,待改进
    for ch in seed:
        if ch == '1':
            return True
    return False


def seedtovector(seed, X):  # 个体转矩阵   got it!     返回ans矩阵

    # temp = np.array(X)
    # ans = np.empty(shape= 0)
    flag = 1
    cnt = 0
    ans = X[:, 0]
    # print(ans)
    for ch in seed:
        if ch == '1':
            if (flag == 1):  # 标记第一个位置
                ans = X[:, cnt]  # 然后初始化第一列
                flag = 2
                print(ans)

            # XX.join(X[cnt])  # = X[cnt]  #  对应位置价加到新矩阵中
            else:
                ans = np.column_stack((ans, X[:, cnt]))  # 在ans后添加一列


        elif ch == '0':  # 过滤掉
            # print("0")
            continue
        cnt = cnt + 1  # cnt计数器

    if (flag == 1):  # 说明seed是全0
        return []
        # return X, []

    # return X, ans, len(ans[0])
    return ans, cnt


#### 个体变异
def Creat_child(XX):
    # 创建一个个体
    bound = math.pow(2, testSDP.GetDataSetSize(XX))
    print("bound:", bound)
    childID = int(np.random.random() * bound)
    if childID == 0:
        childID = 1
    print(childID)
    # 整数转二进制
    child = "{0:b}".format(childID)
    print(child)
    return child


# if __name__ == '__main__':
#     seed = "100100001110100010100111000001101100"  # 38898659436
#     seed1 = "111010011111111101100011110101111111"  # 62813257087
#     print("{0:b}".format(27308878553))  # 1010
#     np.random.rand(10000)
