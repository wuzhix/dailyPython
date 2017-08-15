'''
redis基础操作
'''

# !/usr/bin/python3
# -*- coding: UTF-8 -*-

import redis
import json

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
value = {'1': '2', '11': '22'}
print(value)
print(type(value))
json_value = json.dumps(value)
print(json_value)
print(type(json_value))
r.hset(key, field, json_value)
byte_value = r.hget(key, field)
print(byte_value)
print(type(byte_value))
str_value = byte_value.decode()
print(str_value)
print(type(str_value))
value = json.loads(str_value)
print(value)
print(type(value))


