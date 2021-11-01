# -*- coding: utf-8 -*-
import random
import json
import numpy as np

filename = "./ins_exist_200.npy"
instances = []
for i in range(200):
    ver_serial_num = random.randint(0, 5)
    instances.append(ver_serial_num)
instances = np.array(instances)
print(instances)
np.save(filename, instances)

print(np.load(filename))