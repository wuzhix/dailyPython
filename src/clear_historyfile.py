'''
清理历史文件
该方法可以像shell一样，添加crontab任务
crontab -e
0 2 * * * /usr/bin/python3 /home/clear_historyfile.py /data 7 >>/data/clear_historyfile.log 2>&1
'''

# !/usr/bin/python3
# -*- coding: UTF-8 -*-

import os
import time
import sys


def clear(file, day):
    '''
    :param file: 要清理的目录或文件
    :param day: 间隔天数大于等于day的文件清理掉
    :return:
    '''
    # 判断文件是否为目录
    is_dir = os.path.isdir(file)
    if is_dir:
        # 确保目录结尾是/
        if file[-1] != '/':
            file += '/'
        # 遍历目录
        dir_list = os.listdir(file)
        for path in dir_list:
            # 递归调用清除接口
            clear(file + path, day)
    else:
        history_time = time.time() - day * 24 * 60 * 60
        # 获取文件最近修改时间
        last_modify = os.stat(file).st_mtime
        # print(last_modify)
        last_modify_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_modify))
        # 最近修改时间离当前时间超过day天，清除；否则，不请除
        if last_modify <= history_time:
            # os.remove(file)
            print("file : %s, last modify : %s, will clear" % (file, last_modify_date))
        else:
            print("file : %s, last modify : %s, not clear" % (file, last_modify_date))


if __name__ == '__main__':
    file = "./" if len(sys.argv) < 2 else sys.argv[1]
    day = 7 if len(sys.argv) < 3 else sys.argv[2]
    clear(file, day)
