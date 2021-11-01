# -*- coding: utf-8 -*-
import random
import time
import math
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class PSO(object):
    def __init__(self, exist_ins, mat_diversity):
        self.iter_max = 20  # 迭代数目
        self.num = 200  # 粒子数目
        self.add_ins_num = 40  # autu-scale数
        self.exist_ins = exist_ins # 城市的位置坐标
        # 计算距离矩阵
        self.mat_diversity = mat_diversity
        # 初始化所有粒子
        self.particals = self.random_init(self.num, self.add_ins_num)
        self.degrees = self.compute_individuals(self.particals)
        # 得到初始化群体的最优解
        init_l = min(self.degrees)
        init_index = self.degrees.index(init_l)
        init_individual = self.particals[init_index]
        # 画出初始的路径图
        # init_show = self.exist_ins[init_individual]
        # 记录每个个体的当前最优解
        self.local_best = self.particals
        self.local_best_len = self.degrees
        # 记录当前的全局最优解,长度是iteration
        self.global_best = init_individual
        self.global_best_len = init_l
        # 输出解
        self.best_l = self.global_best_len
        self.best_individual = self.global_best
        # 存储每次迭代的结果，画出收敛图
        self.iter_x = [0]
        self.iter_y = [10000 / init_l]
        
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
        return result

    # 计算一条路径长度
    def compute_diversity_degree(self, individual, mat_diversity):
        # print(individual)
        result = 0.0
        for i in individual:
            for j in exist_ins:
                result += mat_diversity[i][j]
        for i in range(len(individual) - 1):
            for j in range(i + 1, len(individual)):
                result += mat_diversity[individual[i]][individual[j]]
        return result

    # 计算一个群体的长度
    def compute_individuals(self, individuals):
        result = []
        for one in individuals:
            length = self.compute_diversity_degree(one, self.mat_diversity)
            result.append(length)
        return result

    # 评估当前的群体
    def eval_particals(self):
        min_degree = min(self.degrees)
        min_index = self.degrees.index(min_degree)
        cur_individual = self.particals[min_index]
        # 更新当前的全局最优
        if min_degree < self.global_best_len:
            self.global_best_len = min_degree
            self.global_best = cur_individual
        # 更新当前的个体最优
        for i, l in enumerate(self.degrees):
            if l < self.local_best_len[i]:
                self.local_best_len[i] = l
                self.local_best[i] = self.particals[i]

    # 粒子交叉
    def cross(self, cur, best):
        one = cur.copy()
        l = [t for t in range(self.add_ins_num)]
        t = np.random.choice(l,2)
        x = min(t)
        y = max(t)
        cross_part = best[x:y]
        tmp = []
        for i in l:
            if i in range(x, y):
                continue
            tmp.append(one[i])
        # 两种交叉方法
        one = tmp + cross_part
        l1 = self.compute_diversity_degree(one, self.mat_diversity)
        one2 = cross_part + tmp
        l2 = self.compute_diversity_degree(one2, self.mat_diversity)
        if l1 < l2:
            return one, l1
        else:
            return one, l2

    # 粒子变异，路径中某两个节点进行交换
    def mutate(self, one):
        one = one.copy()
        mutate_index = random.choice(range(len(one)))
        one[mutate_index] = random.randint(0, 5)
        l2 = self.compute_diversity_degree(one,self.mat_diversity)
        return one, l2

    # 迭代操作
    def pso(self):
        for cnt in range(1, self.iter_max):
            # 更新粒子群
            for i, one in enumerate(self.particals):
                tmp_l = self.degrees[i]
                # 与当前个体局部最优解进行交叉
                new_one, new_l = self.cross(one, self.local_best[i])
                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_individual = one
                # 如果新产生的比原有的好，或是有一定的概率，新的替换原有的
                if new_l < tmp_l or np.random.rand() < 0.1:
                    one = new_one
                    tmp_l = new_l

                # 与当前全局最优解进行交叉
                new_one, new_l = self.cross(one, self.global_best)
                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_individual = one
                if new_l < tmp_l or np.random.rand()<0.1:
                    one = new_one
                    tmp_l = new_l

                # 变异
                one, tmp_l = self.mutate(one)
                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_individual = one
                if new_l < tmp_l or np.random.rand()<0.1:
                    one = new_one
                    tmp_l = new_l

                # 更新该粒子
                self.particals[i] = one
                self.degrees[i] = tmp_l
            # 评估粒子群，更新个体局部最优和个体当前全局最优
            self.eval_particals()
            # 更新输出解
            if self.global_best_len < self.best_l:
                self.best_l = self.global_best_len
                self.best_individual = self.global_best
            print(cnt, 10000 / self.best_l)
            self.iter_x.append(cnt)
            self.iter_y.append(10000 / self.best_l)
        return self.best_l, self.best_individual

    def run(self):
        best_length, best_individual = self.pso()
        # 画出最终路径
        return self.best_individual, best_length


start_t =time.time()
mat_diversity = np.load("CVE/mat_diversity.npy")
mat_diversity = mat_diversity / np.max(mat_diversity)
exist_ins = np.load("CVE/ins_exist.npy")
# 加上一行因为会回到起点
# show_exist_ins = np.vstack([exist_ins, exist_ins[0]])
# add_ins_num, exist_ins, mat_diversity
model = PSO(exist_ins=exist_ins.copy(), mat_diversity=mat_diversity.copy())
model.run()

end_t =time.time()
print('Running time: %s Seconds'%(end_t-start_t))

iterations = model.iter_x
best_record = model.iter_y
plt.plot(iterations, best_record)
plt.title('Convergence Curve of PSO')
plt.show()

# np.save("data/E_PSO.npy", best_record)

# Best_individual = np.vstack([Best_individual, Best_individual[0]])
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


