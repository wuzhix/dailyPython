'''
求1<=i<=10**12范围内所有d(i)的和的末12位，d(i)表示i的正约数的和，i为整数
'''

# !/usr/bin/python3
# -*- coding: UTF-8 -*-

# 1
# 1   2
# 1       3
# 1   2       4
# 1               5
# 1   2   3           6
# 1                       7
# 1   2       4               8
# 1       3                       9
# 1   2           5                   10
# ...

n = 10 ** 1
m = 1 + int(n ** 0.5)
s = 0
for i in range(1, m):
    # s += int(n / i) * i + int((int(n / i) + m) * (int(n / i) - m + 1) / 2)
    s += n // i * i + (n // i + m) * (n // i - m + 1) // 2
    print(s)
print(s)


