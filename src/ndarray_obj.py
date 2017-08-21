# -*- coding: UTF-8 -*-

import numpy as np

a = np.array([1, 2, 3, 4])
print(a)
print(a.shape, a.dtype)
# 将数组转换为2行
a = np.reshape(a, [2, -1])
print(a)
print(a.shape, a.dtype)
b = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(b)
print(b.shape, b.dtype)
# 将数组转换为1行
b = np.reshape(b, -1)
print(b)
print(b.shape, b.dtype)
# 生成起始值1，终止值10，间隔为2的数组
c = np.arange(1, 10, 2)
print(c)
print(c.shape, c.dtype)
print(c[1], c[0:2], c[:2], c[:-1], c[0:4:3])
d = np.arange(0, 60, 10).reshape(-1, 1) + np.arange(0, 6)
print(d)
print(d[0, 3:5], d[4:, 4:], d[:, 2], d[2::2, ::2])
persontype = np.dtype({
    'names': ['name', 'age', 'weight'],
    'formats': ['S32', 'i', 'f']})
e = np.array([("Zhang", 32, 75.5), ("Wang", 24, 65.2)],
             dtype=persontype)
print(e, e.dtype)
print(e[0]['age'], e[1]['weight'])
