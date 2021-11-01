# -*- coding: utf-8 -*-
import random
import math
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DriodSansMono']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class GA(object):
    # num_generation种群数量
    def __init__(self, add_ins_num, num_generation, iteration, exist_ins, mat_diversity):
        self.add_ins_num = add_ins_num
        self.num_generation = num_generation
        self.scores = []
        self.iteration = iteration
        self.exist_ins = exist_ins
        self.ga_choose_ratio = 0.2      # 0.2
        self.mutate_ratio = 0.05
        # fruits中存每一个个体是下标的list
        self.mat_diversity = mat_diversity
        self.fruits = self.random_init(num_generation, add_ins_num)
        # 显示初始化后的最佳路径
        scores = self.compute_adp(self.fruits)
        sort_index = np.argsort(-scores)
        init_best = self.fruits[sort_index[0]]

        # 存储每个iteration的结果，画出收敛图
        self.iter_x = [0]
        self.iter_y = [1. / scores[sort_index[0]]]

    # 生成一代随机种群
    def random_init(self, num_generation, add_ins_num):
        result = []
        while len(result) <= num_generation:
            # print('while1')
            tmp_individual = []
            for j in range(add_ins_num):
                tmp_individual.append(random.randint(0, 5))
            # tmp_individual.sort()
            if tmp_individual in result:
                continue
            else:
                result.append(tmp_individual)
            result = list(set([tuple(t) for t in result]))
        result = [list(t) for t in result]
        # result.sort()
        # print(result)
        return result

    # 计算一条路径长度
    def compute_diversity_degree(self, individual, mat_diversity):
        result = 0.0
        for i in individual:
            for j in exist_ins:
                result += mat_diversity[i][j]
        for i in range(len(individual)-1):
            for j in range(i+1, len(individual)):
                result += mat_diversity[individual[i]][individual[j]]
        return result

    # 计算种群适应度
    def compute_adp(self, fruits):
        adp = []
        for fruit in fruits:
            degree = self.compute_diversity_degree(fruit, self.mat_diversity)
            adp.append(1.0 / degree)
        return np.array(adp)

    def ga_cross(self, x, y):
        if x == y:
            print("x/y is not satisfied! x = y =", x, y)
        len_ = len(x)
        assert len(x) == len(y)

        cross_len = random.randint(1, len_)
        start_x = random.randint(0,len_-cross_len)
        end_x = start_x + cross_len

        start_y = random.randint(0,len_-cross_len)
        end_y = start_y + cross_len

        # while x[start_x:end_x] == y[start_y:end_y]:
        #     start_y = random.randint(0, len_ - cross_len)
        #     end_y = start_y + cross_len

        # 交叉
        tmp = x[start_x:end_x].copy()
        x[start_x:end_x] = y[start_y:end_y]
        y[start_y:end_y] = tmp
        # x.sort()
        # y.sort()
        return list(x), list(y)

    # 每一代筛选出强的个体成为父代
    def ga_parent(self, scores, ga_choose_ratio):
        # np.argsort(-scores)返回的是根据数值进行排序后的索引列表
        sort_index = np.argsort(-scores).copy()
        # 只取得分排名高的ga_choose_ratio规定的比例
        sort_index = sort_index[0:int(ga_choose_ratio * len(sort_index))]
        parents = []
        parents_score = []
        for index in sort_index:
            parents.append(self.fruits[index])
            parents_score.append(scores[index])
        return parents, parents_score

    # 被用于gene_x, gene_y = self.ga_choose(parents_score, parents)
    def ga_choose(self, genes_score, genes_choose):
        sum_score = sum(genes_score)
        score_ratio = [sub * 1.0 / sum_score for sub in genes_score]
        rand1 = np.random.rand()
        rand2 = np.random.rand()
        for i, sub in enumerate(score_ratio):
            if rand1 >= 0:
                rand1 -= sub
                if rand1 < 0:
                    index1 = i
            if rand2 >= 0:
                rand2 -= sub
                if rand2 < 0:
                    index2 = i
            if rand1 < 0 and rand2 < 0:
                break
        while index1 == index2:
            rand2 = np.random.rand()
            for i, sub in enumerate(score_ratio):
                if rand2 >= 0:
                    rand2 -= sub
                    if rand2 < 0:
                        index2 = i
        # print(index1, index2)
        return list(genes_choose[index1]), list(genes_choose[index2])

    def ga_mutate(self, gene):
        mutate_index = random.choice(range(len(gene)))
        gene[mutate_index] = random.randint(0, 5)
        # gene.sort()
        return list(gene)

    def ga(self):
        # 获得优质父代
        scores = self.compute_adp(self.fruits)
        # 选择部分优秀个体作为父代候选集合
        parents, parents_score = self.ga_parent(scores, self.ga_choose_ratio)
        tmp_best_one = parents[0]
        tmp_best_score = parents_score[0]
        # 新的种群fruits
        fruits = parents.copy()
        # 生成新的种群
        while len(fruits) < self.num_generation:
            # 轮盘赌方式对父代进行选择
            gene_x, gene_y = self.ga_choose(parents_score, parents)
            # 交叉
            gene_x_new, gene_y_new = self.ga_cross(gene_x, gene_y)
            # 变异
            if np.random.rand() < self.mutate_ratio:
                gene_x_new = self.ga_mutate(gene_x_new)
            if np.random.rand() < self.mutate_ratio:
                gene_y_new = self.ga_mutate(gene_y_new)

            # sim = 0
            # while (gene_x_new in fruits) and (gene_y_new in fruits):
            #     sim = sim + 1
            #     print('gene_x_new/gene_y_new are both in fruits * ', sim)
            #     gene_x_new = self.ga_mutate(gene_x_new)
            #     gene_y_new = self.ga_mutate(gene_y_new)

            x_adp = 1. / self.compute_diversity_degree(gene_x_new, self.mat_diversity)
            y_adp = 1. / self.compute_diversity_degree(gene_y_new, self.mat_diversity)
            # 将适应度高的放入种群中
            # print(not gene_x_new in fruits)
            # print(not gene_y_new in fruits)
            if x_adp > y_adp and (not gene_x_new in fruits):
                fruits.append(gene_x_new)
            elif x_adp <= y_adp and (not gene_y_new in fruits):
                fruits.append(gene_y_new)
            # print(len(fruits))
        self.fruits = fruits
        return tmp_best_one, tmp_best_score

    def run(self):
        BEST_LIST = None
        best_score = -math.inf
        self.best_record = []
        for i in range(1, self.iteration + 1):
            tmp_best_one, tmp_best_score = self.ga()
            self.iter_x.append(i)
            self.iter_y.append(1. / tmp_best_score)
            if tmp_best_score > best_score:
                best_score = tmp_best_score
                # BEST_LIST = [t+1 for t in tmp_best_one]
                BEST_LIST = tmp_best_one
            self.best_record.append(10000*best_score)
            print(i, BEST_LIST, 1./best_score)
            # print(i, 10000 * best_score)
        print(10000 * best_score)
        return BEST_LIST, 10000 * best_score


start_t =time.time()
mat_diversity = np.load("CVE/mat_diversity.npy")
mat_diversity = mat_diversity / np.max(mat_diversity)
versions = ['Ubuntu 16', 'Debian 10', 'Busybox', 'Alpine', 'CentOS 7', 'Fedora 30']

exist_ins = np.load("CVE/ins_exist.npy")
# exist_ins = exist_ins[:, 1:]
Best, Best_path = math.inf, None

# add_ins_num, num_generation, iteration, exist_ins, mat_diversity
model = GA(add_ins_num=100, num_generation=25, iteration=20, exist_ins=exist_ins.copy(), mat_diversity=mat_diversity.copy())
# model = GA(add_ins_num=100, num_generation=25, iteration=50, exist_ins=exist_ins.copy(), mat_diversity=mat_diversity.copy())
new_instances, ins_set_diversity = model.run()
if ins_set_diversity < Best:
    Best = ins_set_diversity
    Best_ins_set = new_instances

end_t =time.time()
print('Running time: %s Seconds'%(end_t-start_t))
# # 加上一行因为会回到起点
# fig, axs = plt.subplots(1, 1, sharex=False, sharey=False)
# axs[0].scatter(Best_ins_set[:, 0], Best_ins_set[:,1])
# Best_ins_set = np.vstack([Best_ins_set, Best_ins_set[0]])
# axs[0].plot(Best_ins_set[:, 0], Best_ins_set[:, 1])
# axs[0].set_title('规划结果')

iterations = range(model.iteration)
best_record = model.best_record
plt.plot(iterations, best_record)
plt.title('Convergence Curve of GA')
plt.show()

# np.save("data/E_GA.npy", best_record)