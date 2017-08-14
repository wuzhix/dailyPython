'''
%y 两位数的年份表示（00-99）
%Y 四位数的年份表示（000-9999）
%m 月份（01-12）
%d 月内中的一天（0-31）
%H 24小时制小时数（0-23）
%I 12小时制小时数（01-12）
%M 分钟数（00=59）
%S 秒（00-59）
%a 本地简化星期名称
%A 本地完整星期名称
%b 本地简化的月份名称
%B 本地完整的月份名称
%c 本地相应的日期表示和时间表示
%j 年内的一天（001-366）
%p 本地A.M.或P.M.的等价符
%U 一年中的星期数（00-53）星期天为星期的开始
%w 星期（0-6），星期天为星期的开始
%W 一年中的星期数（00-53）星期一为星期的开始
%x 本地相应的日期表示
%X 本地相应的时间表示
%Z 当前时区的名称
%% %号本身
'''

# !/usr/bin/python3
# -*- coding: UTF-8 -*-

import time

# 获取当前时间戳
cur_time = time.time()
print("当前时间戳为：%.2f" % cur_time)
# 获取本地时间
local_time = time.localtime(cur_time)
print("本地时间为：", local_time)
# 获取本地格式化时间
local_format_time = time.asctime(local_time)
print("本地格式化时间：", local_format_time)
# 时间戳转换为日期格式
cur_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(cur_date)
# 日期格式转换为时间戳
cur_time = time.mktime(time.strptime(cur_date, "%Y-%m-%d %H:%M:%S"))
print(cur_time)

