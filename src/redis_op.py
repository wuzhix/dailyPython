'''
redis基础操作
'''

# !/usr/bin/python3
# -*- coding: UTF-8 -*-

import redis

host = '127.0.0.1'
port = 6379
db = 0

# 连接redis
r = redis.Redis(host=host, port=port, db=db)
key = 'string_key'
value = r.get(key)
if value is None:
    value = 'string_value'
    r.set(key, value)

key = 'hash_key'
field = 'field_1'
value = r.hget(key, field)
if value is None:
    value = 'hash_value'
    r.hset(key, field, value)
