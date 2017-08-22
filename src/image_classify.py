# -*- coding: UTF-8 -*-

import requests
import numpy as np
from PIL import Image
from io import BytesIO
# from skimage import io, transform
import pymysql
import tensorflow as tf
import time
import random
import redis
import json
import os

# 定义图像尺寸
width = 100
height = 100
channel = 3
# 定义redis连接
host = '127.0.0.1'
redis_port = 6379
redis_db = 0
redis_key = 'pic:cat'
# 定义数据库链接
server = '127.0.0.1'
username = 'root'
password = ''
db_port = 3306
database = 'test'
table = 'pic_data'
# 批量运算的数量
# batch = 50


def read_cat():
    r = redis.Redis(host=host, port=redis_port, db=redis_db)
    cat_data = r.get(redis_key)
    if cat_data is None:
        # 连接数据库，这里一定要加上charset="utf8",不然会乱码
        db = pymysql.connect(host=server, port=db_port, user=username, password=password, database=database,
                             charset="utf8")
        # 使用cursor()方法获取操作游标
        cursor = db.cursor()
        # 查询语句
        query_sql = "select distinct(cat) from %s" % table
        # 执行sql语句
        cursor.execute(query_sql)
        # 获取查询结果
        results = cursor.fetchall()
        cat_data = [y for x in results for y in x]
        r.set(redis_key, json.dumps(cat_data))
    else:
        cat_data = cat_data.decode()
        cat_data = json.loads(cat_data)
    return cat_data

cat_arr = read_cat()
start_ids = {}

def read_image():
    if len(start_ids) == 0:
        for cat in cat_arr:
            start_ids[cat] = 1
    # 连接数据库，这里一定要加上charset="utf8",不然会乱码
    db = pymysql.connect(host=server, port=db_port, user=username, password=password, database=database, charset="utf8")
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    '''
    数据库表格
    CREATE TABLE `pic_data` (
      `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT COMMENT '图片id',
      `pic` varchar(255) NOT NULL DEFAULT '' COMMENT '图片url',
      `cat` int(10) unsigned NOT NULL DEFAULT '0' COMMENT '图片分类',
      PRIMARY KEY (`id`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8;
    '''
    results = []
    for cat in start_ids:
        # 查询语句
        query_sql = "select * from %s where id >= %d and cat = %d limit 1" % (table, start_ids[cat], cat)
        # 执行sql语句
        cursor.execute(query_sql)
        # 获取查询结果
        result = cursor.fetchone()
        results.append(result)
        # print(query_sql)
        # print(result, cat, start_ids[cat])
        if result[0] == start_ids[cat]:
            start_ids[cat] += 1
        else:
            start_ids[cat] = result[0] + 1
    # 关闭连接
    db.close()
    imgs = []
    lables = []
    for i, row in enumerate(results):
        # requests.get(url)和Image.open(BytesIO(response.content))都会抛出异常
        try:
            # get请求图片url
            response = requests.get(row[1])
        except:
            continue
        # 图片请求能正常响应
        if response.status_code == 200:
            # 获取网络图片
            try:
                # 加载图片
                img = Image.open(BytesIO(response.content))
                # img = Image.open('1.jpg')
                # 图片尺寸设置为100*100，然后转换为np数组，每个值除以255取浮点数结果
                arr = np.array(img.resize((width, height))) / 255
                # print(row[0], arr.shape)
                # print(arr.shape, arr.dtype)
                # img = io.imread(BytesIO(response.content))
                # arr = transform.resize(img, (width, height))
                # print(arr.shape, arr.dtype)
            except:
                continue
            # yield arr, row[2]
            # 填充list
            if len(arr.shape) == 3 and arr.shape[2] == 3:
                imgs.append(arr)
                lables.append(cat_arr.index(row[2]))
    # asarray将list转换为ndarray
    return np.asarray(imgs, np.float32), np.asarray(lables, np.int32)


# -----------------构建网络----------------------
# 占位符
x = tf.placeholder(tf.float32, shape=[None, width, height, channel], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

# 第一个卷积层（100——>50)
conv1 = tf.layers.conv2d(
    inputs=x,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# 第二个卷积层(50->25)
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# 第三个卷积层(25->12)
conv3 = tf.layers.conv2d(
    inputs=pool2,
    filters=128,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

# 第四个卷积层(12->6)
conv4 = tf.layers.conv2d(
    inputs=pool3,
    filters=128,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

re1 = tf.reshape(pool4, [-1, 6 * 6 * 128])

# 全连接层
dense1 = tf.layers.dense(inputs=re1,
                         units=1024,
                         activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
dense2 = tf.layers.dense(inputs=dense1,
                         units=512,
                         activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
logits = tf.layers.dense(inputs=dense2,
                         units=len(cat_arr),
                         activation=None,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
# ---------------------------网络结束---------------------------

loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 训练和测试数据，可将n_epoch设置更大一些
n_epoch = 1000
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for epoch in range(n_epoch):
    start_time = time.time()

    # training
    train_loss, train_acc, n_batch = 0, 0, 0

    x_train_a, y_train_a = read_image()
    # print(y_train_a)
    _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
    train_loss += err
    train_acc += ac
    n_batch += 1
    logit = sess.run(logits, feed_dict={x: x_train_a})
    # print(logit)
    # print(y_train_a)
    print("   train loss: %f" % (train_loss / n_batch))
    print("   train acc: %f" % (train_acc / n_batch))

sess.close()
