'''
目录和文件操作
'''

# !/usr/bin/python3
# -*- coding: UTF-8 -*-

import os

# 获取当前目录
cur_dir = os.getcwd()
print(cur_dir)
# 将路径分解为目录名和文件名
src = cur_dir+'\\1.txt'
fpath , fname = os.path.split(src)
print(fpath)
print(fname)
# 分解文件名的扩展名
fpath_name, ftext = os.path.splitext(src)
print(fpath_name)
print(ftext)
# 判断路径是否存在
is_exists = os.path.exists(src)
print(is_exists)
# 获取目录列表
path_list = os.listdir(fpath)
print(path_list)
# 文件操作和c语言类似，就不列出来了

