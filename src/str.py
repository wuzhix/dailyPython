'''
字符串操作
'''

# !/usr/bin/python3
# -*- coding: UTF-8 -*-

str = 'ab,cde,fg'
# 查找字符串
pos = str.find('a')
print('a index is ', pos)
pos = str.find('de')
print('de index is ', pos)
pos = str.find('m')
print('m index is ', pos)
# 小写转大写
upper = str.upper()
print(upper)
# 大写转小写
lower = upper.lower()
print(lower)
# 字符串翻转
rev = str[::-1]
print(rev)
# 字符串分隔
split = str.split(',')
print(split)
# 连接字符串
delimiter = ','
mylist = ['Brazil', 'Russia', 'India', 'China']
print(delimiter.join(mylist))
