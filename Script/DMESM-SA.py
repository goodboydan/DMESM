# -*- coding: utf-8 -*-
import random
import math
import time
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class SA(object):
    def __init__(self, exist_ins, mat_diversity):
        self.T0 = 4000
        self.Tend = 1e-3
        self.rate = 0.97
        self.ind_num = 100
        self.add_ins_num = 50  # autu-scale数
        self.exist_ins = exist_ins # 城市的位置坐标
        # 计算距离矩阵
        self.mat_diversity = mat_diversity
        self.scores = []
        self.fires = []
        self.fire = self.random_init(self.ind_num, self.add_ins_num)
        # 显示初始化后的路径
        init_degree = 1. / self.compute_diversity_degree(self.fire, self.mat_diversity)
        # init_best = self.location[self.fire]
        # 存储存储每个温度下的最终路径，画出收敛图
        self.iter_x = [0]
        self.iter_y = [10000 * init_degree]

    # 随机初始化
    def random_init(self, num_generation, add_ins_num):
        result = []
        while len(result) <= num_generation:
            tmp_individual = []
            for j in range(add_ins_num):
                tmp_individual.append(random.randint(0, 5))
            if tmp_individual in result:
                continue
            else:
                result.append(tmp_individual)
            result = list(set([tuple(t) for t in result]))
        result = [list(t) for t in result]
        # 和前几个算法不同的是，前几个算法输出的是多组结果，这个算法里输出的是最好的一个结果
        degrees = self.compute_individuals(result)
        sortindex = np.argsort(degrees)
        index = sortindex[0]
        return result[index]

    # 计算一条路径长度
    def compute_diversity_degree(self, individual, mat_diversity):
        # print(individual)
        result = 0.0
        for i in individual:
            for j in self.exist_ins:
                result += mat_diversity[i][j]
        for i in range(len(individual) - 1):
            for j in range(i + 1, len(individual)):
                result += mat_diversity[individual[i]][individual[j]]
        return result

    # 计算一个温度下产生的一个群体的长度
    def compute_individuals(self, individuals):
        result = []
        for one in individuals:
            length = self.compute_diversity_degree(one, self.mat_diversity)
            result.append(length)
        return result

    # 产生一个新的解：随机生成新的元素
    def get_new_fire(self, fire):
        fire = fire.copy()
        t = [x for x in range(len(fire))]
        a, b = np.random.choice(t, 2)
        # fire[a:b] = [random.randint(0, 5) for i in range(len(fire[a:b]))]
        # a, b, c = np.random.choice(t, 3)
        fire[a] = random.randint(0, 5)
        fire[b] = random.randint(0, 5)
        # fire[c] = random.randint(0, 5)
        return fire

    # 退火策略，根据温度变化有一定概率接受差的解
    def eval_fire(self, raw, get, temp):
        degree1 = self.compute_diversity_degree(raw, self.mat_diversity)
        degree2 = self.compute_diversity_degree(get, self.mat_diversity)
        dc = degree2 - degree1
        p = max(1e-1, np.exp(-dc / temp))
        # 如果新解更好则接受新解
        if degree2 < degree1:
            return get, degree2
        # 如果新解不如旧解，也以一定的概率接收新解
        elif np.random.rand() <= p:
            return get, degree2
        else:
            return raw, degree1

    # 模拟退火总流程
    def sa(self):
        count = 0
        # 记录最优解
        best_individual = self.fire
        best_degree = self.compute_diversity_degree(self.fire, self.mat_diversity)

        while self.T0 > self.Tend:
            count += 1
            # 产生在这个温度下的随机解
            tmp_new = self.get_new_fire(self.fire.copy())
            # 根据温度判断是否选择这个解
            self.fire, fire_degree = self.eval_fire(best_individual, tmp_new, self.T0)
            # 更新最优解
            if fire_degree < best_degree:
                best_degree = fire_degree
                best_individual = self.fire
            # 降低温度
            self.T0 *= self.rate
            # 记录路径收敛曲线
            self.iter_x.append(count)
            self.iter_y.append(10000 / best_degree)
            print(count, 10000 / best_degree)
        return best_degree, best_individual

    def run(self):
        best_degree, best_individual = self.sa()
        return best_individual, best_degree


start_t =time.time()
mat_diversity = np.load("CVE/mat_diversity.npy")
mat_diversity = mat_diversity / np.max(mat_diversity)
exist_ins = np.load("CVE/ins_exist.npy")
Best, Best_individual = math.inf, None

model = SA(exist_ins=exist_ins.copy(), mat_diversity=mat_diversity.copy())
individual, ind_degree = model.run()

print(10000 / ind_degree)
end_t =time.time()
print('Running time: %s Seconds'%(end_t-start_t))

if ind_degree < Best:
    Best = ind_degree
    Best_individual = individual
# 加上一行因为会回到起点
# Best_individual = np.vstack([Best_individual, Best_individual[0]])
iterations = model.iter_x
best_record = model.iter_y
plt.plot(iterations, best_record)
plt.title('Convergence Curve of SA')
plt.show()

np.save("data/E_SA.npy", best_record)
# fig, axs = plt.subplots(2, 1, sharex=False, sharey=False)
# axs[0].scatter(Best_individual[:, 0], Best_individual[:,1])
# Best_individual = np.vstack([Best_individual, Best_individual[0]])
# axs[0].plot(Best_individual[:, 0], Best_individual[:, 1])
# axs[0].set_title('规划结果')
# iterations = model.iter_x
# best_record = model.iter_y
# axs[1].plot(iterations, best_record)
# axs[1].set_title('收敛曲线')
# plt.show()

