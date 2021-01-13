import utils.MOEAD_Utils as MOEAD_Utils
import utils.Draw_Utils as Draw_Utils
import numpy as np
import numpy as np
import math
import testSDP
import datetime

'''
遗传算法工具包
'''


def Creat_child(moead):
    # 创建一个个体
    dataSetSize = testSDP.GetDataSetSize(moead.XX)
    bound = math.pow(2, dataSetSize)
    print("bound:", bound)
    childID = int(np.random.random() * bound)
    # childID = 222  # fisher下一样
    if childID == 0:
        childID = 1
    print(childID)
    # 整数转二进制
    child = "{0:0{1:}b}".format(childID, dataSetSize)
    print(child)
    return child


def Creat_Pop(moead):
    # 创建moead.Pop_size个种群
    Pop = []
    Pop_FV = []
    if moead.Pop_size < 1:
        print('error in creat_Pop')
        return -1
    while len(Pop) != moead.Pop_size:
        print("seed id the ", len(Pop), "st ")
        print("XX原始数据集：", moead.XX)
        seed = Creat_child(moead.XX)
        print(seed)
        print(type(seed))
        Pop.append(seed)
        Pop_FV.append(
            moead.Test_fun.Func(seed, moead.XX.XX, moead.XX.y, moead.XX.method, moead.XX.n_clfs, moead.XX.fs_functions,
                                moead.XX.score_name))  # 计算个体函数值
    moead.Pop, moead.Pop_FV = Pop, Pop_FV

    print(Pop, "Pop_Fv", Pop_FV)  # 测试输出
    return Pop, Pop_FV


def mutate(moead, p1):
    # 突变个体的策略1
    var_num = moead.Test_fun.Dimention
    for i in range(int(var_num * 0.1)):
        d = moead.Test_fun.Bound[0] + (moead.Test_fun.Bound[1] - moead.Test_fun.Bound[0]) * np.random.rand()
        d = d * np.random.randint(-1, 1)
        d = d / 10
        j = np.random.randint(0, var_num, size=1)[0]
        p1[j] = p1[j] + d
    return p1


def mutate2(moead, y1):
    # 突变个体的策略2
    index = int(np.random.rand() * moead.Test_fun.GetDataSetSize(moead.XX.XX))

    return replace_char(y1, index)

    # dj = 0
    # uj = np.random.rand()
    # if uj < 0.5:
    #     dj = (2 * uj) ** (1 / 6) - 1
    # else:
    #     dj = 1 - 2 * (1 - uj) ** (1 / 6)
    # y1 = y1 + dj
    # y1[y1 > moead.Test_fun.Bound[1]] = moead.Test_fun.Bound[1]
    # y1[y1 < moead.Test_fun.Bound[0]] = moead.Test_fun.Bound[0]
    # return y1


def crossover(moead, pop1, pop2):
    # 交叉个体的策略1
    var_num = moead.Test_fun.Dimention
    r1 = int(var_num * np.random.rand())
    if np.random.rand() < 0.5:
        pop1[:r1], pop2[:r1] = pop2[:r1], pop1[:r1]
    else:
        pop1[r1:], pop2[r1:] = pop2[r1:], pop1[r1:]
    return pop1, pop2


def crossover2(moead, y1, y2):
    # 交叉个体的策略2
    var_num = moead.Test_fun.Dimention
    yj = 0
    uj = np.random.rand()

    #
    print(y1, "    ++++++", y2)
    print(type(y1))
    print(type(y2))
    print(uj)
    #
    # y1 = int(y1, 2)
    # y2 = int(y2, 2)
    # y1 = int(y1)
    # y2 = int(y2)
    print(y1, "    ++++++", y2)
    print(type(y1))
    #
    if uj < 0.5:
        yj = (2 * uj) ** (1 / 3)
    else:
        yj = (1 / (2 * (1 - uj))) ** (1 / 3)
    y1 = 0.5 * (1 + yj) * y1 + (1 - yj) * y2
    y2 = 0.5 * (1 - yj) * y1 + (1 + yj) * y2

    print(y1, "    ++++++", y2)

    y1[y1 > moead.Test_fun.Bound[1]] = moead.Test_fun.Bound[1]
    y1[y1 < moead.Test_fun.Bound[0]] = moead.Test_fun.Bound[0]
    y2[y2 > moead.Test_fun.Bound[1]] = moead.Test_fun.Bound[1]
    y2[y2 < moead.Test_fun.Bound[0]] = moead.Test_fun.Bound[0]
    return y1, y2


#### 自定义的
def crossover3(moead, pop1, pop2):
    len_num = moead.Test_fun.GetDataSetSize(moead.XX.XX)
    cut_num = int(len_num * np.random.rand())
    pop1 = np.array(pop1).reshape(1, -1)
    pop2 = np.array(pop2).reshape(1, -1)

    str1 = pop1[0][0]
    str2 = pop2[0][0]

    temp1 = str1[:cut_num] + str2[cut_num:]
    temp2 = str1[cut_num:] + str2[:cut_num]
    # 考虑要不要转ndarry格式
    # temp1 = np.array(temp1)
    # temp2 = np.array(temp2)
    return temp1, temp2


def EO(moead, wi, p1):
    print("p1：：", p1, "   wi::", wi)

    # p1 = p1.reshape(1, -1)  # 后加的
    # m = p1.shape[0]
    m = 1  # 目前自定义为1,或者为其他值
    tp_best = np.copy(p1)

    qbxf_tp = MOEAD_Utils.cpt_tchbycheff(moead, wi, tp_best)
    Up = np.sqrt(moead.Test_fun.Bound[1] - moead.Test_fun.Bound[0]) / 2
    print("qbxf_tp:", qbxf_tp, "Up:", Up, "p1:", p1)
    print("tp_best:", tp_best, type(tp_best))

    h = 0
    for i in range(m):
        if h == 1:
            return tp_best
        # ndarry转字符串
        temp_best = np.copy(p1).reshape(1, -1)
        temp_best = temp_best[0][0]
        # 高斯分布，正态分布
        rd = np.random.normal(0, Up, 1)
        print("temp_best:", temp_best, type(temp_best), "rd=:", rd)
        datasetSize = moead.Test_fun.GetDataSetSize(moead.XX.XX)
        # print("datasetSize = ", datasetSize)
        # do while~ 先跑一遍
        index = int(rd * datasetSize)
        index = abs(index)
        # index不在范围内，使用随机函数生成一下index吧
        while index < 0 or index >= datasetSize:
            index = int(np.random.rand() * datasetSize)

        print("index:", index, type(index))
        temp_best = replace_char(temp_best, index)
        print("tp_best:", tp_best, type(tp_best))
        # temp_best[i] = temp_best[i] + rd
        # temp_best[temp_best > moead.Test_fun.Bound[1]] = moead.Test_fun.Bound[1]
        # temp_best[temp_best < moead.Test_fun.Bound[0]] = moead.Test_fun.Bound[0]

        qbxf_te = MOEAD_Utils.cpt_tchbycheff(moead, wi, temp_best)
        if qbxf_te < qbxf_tp:
            h = 1
        qbxf_tp = qbxf_te
        # tp_best[:] = temp_best[:]
        tp_best = temp_best
    return tp_best


def cross_mutation(moead, p1, p2):  # p1 p2 都是POP中的seed
    y1 = np.copy(p1)
    y2 = np.copy(p2)
    c_rate = 1
    m_rate = 0.5
    if np.random.rand() < c_rate:
        y1, y2 = crossover3(moead, y1, y2)
    if np.random.rand() < m_rate:
        y1 = mutate2(moead, y1)
        y2 = mutate2(moead, y2)
    return y1, y2


def generate_next(moead, gen, wi, p0, p1, p2):
    # 进化下一代个体。基于自身Xi+邻居中随机选择的2个Xk，Xl 还考虑gen 去进化下一代
    qbxf_p0 = MOEAD_Utils.cpt_tchbycheff(moead, wi, p0)
    qbxf_p1 = MOEAD_Utils.cpt_tchbycheff(moead, wi, p1)
    qbxf_p2 = MOEAD_Utils.cpt_tchbycheff(moead, wi, p2)

    qbxf = np.array([qbxf_p0, qbxf_p1, qbxf_p2])
    best = np.argmin(qbxf)
    # 选中切比雪夫距离最小（最好的）个体
    Y1 = [p0, p1, p2][best]
    # 需要深拷贝成独立的一份
    n_p0, n_p1, n_p2 = np.copy(p0), np.copy(p1), np.copy(p2)

    if gen % 10 == 0:
        # 每格10代，有小概率进行EO优化（效果好，但是复杂度高）
        if np.random.rand() < 0.1:
            # if np.random.rand() < 0:
            n_p0 = EO(moead, wi, n_p0)
    # 交叉
    n_p0, n_p1 = cross_mutation(moead, n_p0, n_p1)
    n_p1, n_p2 = cross_mutation(moead, n_p1, n_p2)
    print(n_p0, n_p1, n_p2)
    print(type(n_p0), type(n_p1), type(n_p2))
    # 交叉后的切比雪夫距离
    qbxf_np0 = MOEAD_Utils.cpt_tchbycheff(moead, wi, n_p0)
    qbxf_np1 = MOEAD_Utils.cpt_tchbycheff(moead, wi, n_p1)
    qbxf_np2 = MOEAD_Utils.cpt_tchbycheff(moead, wi, n_p2)

    qbxf = np.array([qbxf_p0, qbxf_p1, qbxf_p2, qbxf_np0, qbxf_np1, qbxf_np2])
    best = np.argmin(qbxf)
    # 选中切比雪夫距离最小（最好的）个体
    Y2 = [p0, p1, p2, n_p0, n_p1, n_p2][best]

    # 随机选中目标中的某一个目标进行判断，目标太多，不要贪心，随机选一个目标就好
    fm = np.random.randint(0, moead.Test_fun.Func_num)
    # 如果是极小化目标求解，以0。5的概率进行更详细的判断。（返回最优解策略不能太死板，否则容易陷入局部最优）
    # if moead.problem_type == 0 and np.random.rand() < 0.5:
    #     # FY1 = moead.Test_fun.Func(Y1)
    #     FY1 = moead.Test_fun.Func(Y1, moead.XX.XX, moead.XX.y, moead.XX.method, moead.XX.n_clfs, moead.XX.fs_functions,
    #                               moead.XX.score_name)
    #     # FY2 = moead.Test_fun.Func(Y2)
    #     FY2 = moead.Test_fun.Func(Y2, moead.XX.XX, moead.XX.y, moead.XX.method, moead.XX.n_clfs, moead.XX.fs_functions,
    #                               moead.XX.score_name)
    #     # 如果随机选的这个目标Y2更好，就返回Y2的
    #     if FY2[fm] < FY1[fm]:
    #         return Y2
    #     else:
    #         return Y1

    # 本文中Z = [0,1],以下部分为本文中的策略
    if np.random.rand() < 0.5:
        FY1 = moead.Test_fun.Func(Y1, moead.XX.XX, moead.XX.y, moead.XX.method, moead.XX.n_clfs, moead.XX.fs_functions,
                                  moead.XX.score_name)
        FY2 = moead.Test_fun.Func(Y2, moead.XX.XX, moead.XX.y, moead.XX.method, moead.XX.n_clfs, moead.XX.fs_functions,
                                  moead.XX.score_name)
        # 如果随机选的这个目标Y2更好，就返回Y2的
        if fm == 0:  # 特征数最少
            if FY2[fm] < FY1[fm]:
                return Y2
            else:
                return Y1
        else:  # AUC值最大化
            if FY2[fm] < FY1[fm]:
                return Y1
            else:
                return Y2

    return Y2


def envolution(moead):
    # 进化，开始进化moead.max_gen轮
    for gen in range(moead.max_gen):
        # 用于图像展示的时候告诉它，现在在第几轮了
        moead.gen = gen
        # 对数组moead.Pop中的每一个个体，开始拿出来，你们要被进化了！
        # 个体序号pi，个体p
        for pi, p in enumerate(moead.Pop):
            # 第pi号个体的邻居集
            Bi = moead.W_Bi_T[pi]
            # 随机选一个T内的数，作为pi的邻居。
            # （邻居你可以想象成：物种，你总不能人狗杂交吧？所以个体pi只能与他的T个前后的邻居权重，管的个体杂交进化）
            # 比如：T=2，权重(0.1,0.9)约束的个体的邻居是：权重(0,1)、(0.2,0.8)约束的个体。永远固定不变
            k = np.random.randint(moead.T_size)
            l = np.random.randint(moead.T_size)
            # 随机从邻居内选2个个体，产生新解
            ik = Bi[k]
            il = Bi[l]
            Xi = moead.Pop[pi]
            Xk = moead.Pop[ik]
            Xl = moead.Pop[il]
            # 进化下一代个体。基于自身Xi+邻居中随机选择的2个Xk，Xl 还考虑gen 去进化下一代
            Y = generate_next(moead, gen, pi, Xi, Xk, Xl)
            # 计算当前Xi，不进化前的切比雪夫距离
            cbxf_i = MOEAD_Utils.cpt_tchbycheff(moead, pi, Xi)
            # 计算当前Xi，进化后的切比雪夫距离。（比较进化更好？那就保留）
            cbxf_y = MOEAD_Utils.cpt_tchbycheff(moead, pi, Y)
            # 不能随随便便一点点好就要了（自己的策略设计）。超过d才更新
            d = 0.001
            # 开始比较是否进化出了更好的下一代，这样才保留
            if cbxf_y < cbxf_i:
                # 用于绘图：当前进化种群中，哪个，被，正在 进化。draw_w=true的时候可见
                moead.now_y = pi
                # 计算下一代的函数值
                # F_Y = moead.Test_fun.Func(Y)[:]
                # 使用布隆过滤器去重
                # F_Y = []
                # if (Y in moead.sbf):
                #     F_Y = [moead.Test_fun.GetDataSetSize(moead.XX.XX), 0]
                # else:
                #     moead.sbf.add(Y)
                F_Y = moead.Test_fun.Func(Y, moead.XX.XX, moead.XX.y, moead.XX.method, moead.XX.n_clfs,
                                          moead.XX.fs_functions,
                                          moead.XX.score_name)[:]

                # 更新函数值到moead.EP_X_FV中。都进化出更好切比雪夫下一代了，自然要更新多目标中的目标的函数值
                MOEAD_Utils.update_EP_By_ID(moead, pi, F_Y)
                # 都进化出更好切比雪夫下一代了，有可能有更好的理想点，尝试更新理想点

                # MOEAD_Utils.update_Z(moead, Y)

                if abs(cbxf_y - cbxf_i) > d:
                    # 超过d才更新。更新支配前沿。红色点那些
                    MOEAD_Utils.update_EP_By_Y(moead, pi)
            MOEAD_Utils.update_BTX(moead, Bi, Y)
        # 是否需要动态展示
        if moead.need_dynamic:
            Draw_Utils.plt.cla()
            if moead.draw_w:
                Draw_Utils.draw_W(moead)
            Draw_Utils.draw_MOEAD_Pareto(moead, moead.name + ",gen:" + str(gen) + "")
            Draw_Utils.plt.pause(0.001)
        print('迭代 %s,支配前沿个体数量len(moead.EP_X_ID) :%s,moead.Z:%s' % (gen, len(moead.EP_X_ID), moead.Z))
        nowTime = str(datetime.datetime.now().strftime('%y-%m-%d %H:%M:%S')) + ' 星期 ' + str(
            datetime.datetime.now().isoweekday())
        print("当前系统时间：", nowTime)
        print("当前代的最优解集，打印输出：")  # 下边for循环中的
        for ID in moead.EP_X_ID:
            # if (moead.Pop_FV[ID] in moead.sbf):
            #     continue
            # else:
            #     moead.sbf.add(moead.Pop_FV[ID])
            #     cnt = cnt + 1
            print("POPSeed:", moead.Pop[ID], "FV结果：", moead.Pop_FV[ID])

    return moead.EP_X_ID


# 在第index位置进行突变
def replace_char(string, index):
    string = list(string)
    if string[index] == '1':
        string[index] = '0'
    else:
        string[index] = '1'
    return ''.join(string)
