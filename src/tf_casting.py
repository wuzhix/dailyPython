'''
tf基础数据转换
'''

# !/usr/bin/python3
# -*- coding: UTF-8 -*-

import tensorflow as tf

sess = tf.InteractiveSession()
s = tf.constant(['123', '257'])
print(s.eval(), s.dtype)
# 字符串转数字
num = tf.string_to_number(s)
print('数字 : ', num.eval(), num.dtype)
# 数字转double
d_num = tf.to_double(num)
print('double : ', d_num.eval(), d_num.dtype)
# 数字转float
f_num = tf.to_float(num)
print('float : ', f_num.eval(), f_num.dtype)
# 数字转bfloat16, bfloat16是什么鬼？网上没查出来
f16_num = tf.to_bfloat16(num)
print('bfloat16 : ', f16_num.eval(), f16_num.dtype)
# 数字转int32
i32_num = tf.to_int32(num)
print('int32 : ', i32_num.eval(), i32_num.dtype)
# 数字转int64
i64_num = tf.to_int64(num)
print('int64 : ', i64_num.eval(), i64_num.dtype)
# 转换为指定类型
cast_num = tf.cast(i64_num, tf.int8)
print('cast : ', cast_num.eval(), cast_num.dtype)
# bitcast转换
bit_num = tf.bitcast(i64_num, tf.int8)
print('bitcast : ', bit_num.eval(), bit_num.dtype)
# saturate_cast转换
saturate_cast = tf.saturate_cast(i64_num, tf.int32)
print('saturate_cast : ', saturate_cast.eval(), saturate_cast.dtype)

