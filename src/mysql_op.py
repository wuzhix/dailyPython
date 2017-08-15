'''
mysql基础操作
'''

# !/usr/bin/python3
# -*- coding: UTF-8 -*-

import pymysql
import json

server = '127.0.0.1'
username = 'root'
password = ''
database = 'test'
table = 'my_user'

# 连接数据库，这里一定要加上charset="utf8",不然会乱码
db = pymysql.connect(host=server, user=username, password=password, database=database, charset="utf8")
# 使用cursor()方法获取操作游标
cursor = db.cursor()
# 如果数据表已经存在使用 execute() 方法删除表。
cursor.execute("drop table if exists %s" % table)
# 创建sql
create_sql = """create table %s(
    `id` int(10) unsigned not null auto_increment comment '主键',
    `sex` tinyint(1) unsigned not null default '0' comment '性别，0女，1男',
    `age` tinyint(3) unsigned not null default '0' comment '年龄',
    primary key (`id`)
    )engine=innodb default charset=utf8 comment '用户表'""" % table
cursor.execute(create_sql)
# ping连接，断开自动重连
db.ping()
# SQL插入语句
insert_sql = "insert into %s (sex,age) values(0,20),(1,19)" % table
try:
   # 执行语句
   cursor.execute(insert_sql)
   # 提交到数据库执行
   db.commit()
except:
   # Rollback in case there is any error
   db.rollback()
db.ping()
# 删除sql
delete_sql = "delete from %s where age = 20" % table
try:
   # 执行语句
   cursor.execute(delete_sql)
   # 提交到数据库执行
   db.commit()
except:
   # Rollback in case there is any error
   db.rollback()
db.ping()
# 更新sql
update_sql = "update %s set age = 20 where age = 19 limit 1" % table
try:
   # 执行语句
   cursor.execute(update_sql)
   # 提交到数据库执行
   db.commit()
except:
   # Rollback in case there is any error
   db.rollback()
db.ping()
# 查询sql
query_sql = "select * from %s limit 100" % table
try:
    # 执行语句
    cursor.execute(query_sql)
    results = cursor.fetchall()
    for i, row in enumerate(results):
        print(i)
        print(row)
except:
    print('error: unable to fetch data')
db.close()




