# -*- coding: utf-8 -*-
# 导入模块
import csv
import numpy as np
import matplotlib.pyplot as plt
import xlrd

# 打开文件方式1：
work_book = xlrd.open_workbook('cve-selected.xls')
# 方式2：
# w2 = xlrd.book.open_workbook_xls('表02.xls')

# 按索引获取sheet对象
sheet_1 = work_book.sheet_by_index(0)

data_col = [sheet_1.col_values(i) for i in range(sheet_1.ncols)]

Ubuntu = []
Debian = []
Busybox = []
Alpine = []
CentOS = []
Fedora = []
ALL = [Ubuntu, Debian, Busybox, Alpine, CentOS, Fedora]


for j in range(len(data_col)):
    # globals()[data_col[j][0]] = []
    for i in range(1, len(data_col[j])):
        if data_col[j][i] is not '':
            ALL[j].append(data_col[j][i])


def compare(os1, os2):
    inter = set(os1).intersection(set(os2))
    return len(inter)


def cal_matrix(lists_list):
    mat = []
    for i in range(len(lists_list)):
        mat_line = []
        os_1 = lists_list[i]
        for j in range(len(lists_list)):
            os_2 = lists_list[j]
            mat_line.append(compare(os_1, os_2))
        mat.append(mat_line)
    return mat


result = np.array(cal_matrix(ALL))
np.save("mat_diversity.npy", result)


b = np.load("mat_diversity.npy")
print(b)