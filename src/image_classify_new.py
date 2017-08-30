'''
CREATE TABLE `goods_cat` (
  `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT COMMENT '自增id',
  `parentid` int(10) NOT NULL DEFAULT '0' COMMENT '父类id',
  `catid` int(10) unsigned NOT NULL COMMENT '分类id',
  `catname` varchar(50) NOT NULL DEFAULT '' COMMENT '分类名称',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE `goods_pic` (
  `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT COMMENT '自增id',
  `sa_id` bigint(20) unsigned NOT NULL COMMENT '图片id',
  `pic` varchar(255) NOT NULL DEFAULT '' COMMENT '图片url',
  `one` smallint(5) unsigned NOT NULL DEFAULT '0' COMMENT '图片一级分类',
  `two` smallint(5) unsigned NOT NULL DEFAULT '0' COMMENT '图片二级分类',
  `three` smallint(5) unsigned NOT NULL DEFAULT '0' COMMENT '图片三级分类',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
'''
# -*- coding: UTF-8 -*-

import requests
import numpy as np
from skimage import io, transform
from io import BytesIO
import pymysql
import tensorflow as tf
import time
import redis
import json
import os
import sys
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 定义图像尺寸，width宽度，height高度，channel通道，RGB为3，黑白为1
width = 100
height = 100
channel = 3
# 定义redis连接
host = '127.0.0.1'
redis_port = 6379
redis_db = 0
redis_key = 'goods_pic:cat'
# 定义数据库链接
server = '127.0.0.1'
username = 'root'
password = ''
db_port = 3306
database = 'test'
table = 'goods_pic'

# 分类list
cat_arr = []
# 每个分类的起始id，用来查找数据
start_ids = {}


# 读取图片分类
def read_cat():
    # 连接redis
    r = redis.Redis(host=host, port=redis_port, db=redis_db)
    # redis取数
    cat_data = r.get(redis_key)
    if cat_data is None:
        # 连接数据库，这里一定要加上charset，不然查询的中文会乱码
        db = pymysql.connect(host=server, port=db_port, user=username, password=password, database=database,
                             charset="utf8")
        # 使用cursor()方法获取操作游标
        cursor = db.cursor()
        # 查出所有不同类别
        query_sql = "select distinct(three) from %s where three != 0" % table
        # 执行sql语句
        cursor.execute(query_sql)
        # 获取查询结果
        results = cursor.fetchall()
        db.close()
        # 将结果转换为一维list
        cat_data = [y for x in results for y in x]
        del results
        # 缓存分类
        r.set(redis_key, json.dumps(cat_data))
    else:
        # 读取分类
        cat_data = cat_data.decode()
        cat_data = json.loads(cat_data)
    return cat_data


# 读取图片
# batch_size 每次查找每个分类的数据条数
# batch_times 每一轮循环批量训练的次数
def read_image(batch_size, batch_times):
    global start_ids, cat_arr
    # 初始化每种分类的图片起始id
    if len(start_ids) == 0:
        for cat in cat_arr:
            start_ids[cat] = 1
    # print(start_ids)
    # 连接数据库，这里一定要加上charset，不然读取的中文会乱码
    db = pymysql.connect(host=server, port=db_port, user=username, password=password, database=database, charset="utf8")
    cur_times = 0
    while True:
        db.ping()
        # 使用cursor()方法获取操作游标
        cursor = db.cursor()
        results = []
        start_time = time.time()
        print('prepare data')
        sys.stdout.flush()
        for cat in start_ids:
            # 查询语句
            query_sql = "select * from %s where id >= %d and three = %d limit %d" \
                        % (table, start_ids[cat], cat, batch_size)
            # 执行sql语句
            cursor.execute(query_sql)
            # 获取查询结果
            result = cursor.fetchall()
            if len(result) > 0:
                results = results + list(result)
                # print(query_sql)
                # print(result, cat, start_ids[cat])
                # 更新每种分类的起始id
                if result[-1][0] == start_ids[cat]:
                    start_ids[cat] += batch_size + 1
                else:
                    start_ids[cat] = result[-1][0] + 1
                del result
        end_time = time.time()
        diff = end_time - start_time
        print('prepare end, count : %d, time : %.2f' % (len(results), diff))
        sys.stdout.flush()
        imgs = []
        lables = []
        # print(len(results))
        for i, row in enumerate(results):
            # requests.get(url)和Image.open(BytesIO(response.content))都会抛出异常
            try:
                # get请求图片url
                url = row[2][:row[2].index('?')]
                response = requests.get(url)
            except:
                continue
            # 图片请求能正常响应
            if response.status_code == 200:
                # 获取网络图片
                try:
                    # 加载图片
                    img = io.imread(BytesIO(response.content))
                    arr = transform.resize(img, (width, height))
                    del img
                    # print(arr.shape, arr.dtype)
                except:
                    continue
                # 填充list
                if len(arr.shape) == 3 and arr.shape[2] == channel:
                    imgs.append(arr)
                    lables.append(cat_arr.index(row[5]))
                del arr
            del response
        # asarray将list转换为ndarray
        del results
        # print('make data')
        # end_time = time.time()
        # diff = round(end_time - start_time, 2)
        # print('make data: ', diff)
        if len(imgs) > 0:
            yield np.asarray(imgs, np.float32), np.asarray(lables, np.int32)
        else:
            start_ids = {}
        del imgs
        del lables
        cur_times += 1
        if cur_times >= batch_times:
            db.close()
            break


# 训练模型
def model():
    global cat_arr
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

    # 损失函数
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
    # 使用Adam 算法的Optimizer
    train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
    # 结果是否匹配
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
    # 计算准确率
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return x, y_, train_op, loss, acc, logits


# 训练数据
# x 图片数据
# y_ 图片分类数据
# train_op优化算法
# loss 损失函数
# 准确率
# n_epoch 循环次数
# batch_size 每个分类读取的图片数
# 每次训练批量迭代次数
# 理论上总共训练图片张数 n_epoch * batch_size * batch_times * len(cat_arr)
def train(x, y_, train_op, loss, acc, n_epoch=10, batch_size=10, batch_times=1000):
    sess = tf.InteractiveSession()

    saver = tf.train.Saver(max_to_keep=1)
    # 读取保存的模型
    model_file = tf.train.latest_checkpoint(table+'/')
    if model_file is None:
        # 如果没有保存模型，初始化所有变量
        sess.run(tf.global_variables_initializer())
    else:
        # 如果保存了模型，直接从模型中初始化变量
        saver.restore(sess, model_file)
    max_acc = 0
    for epoch in range(n_epoch):
        # training
        train_loss, train_acc, n_batch = 0, 0, 0
        # 开始训练（小批量优化算法）
        for x_train_a, y_train_a in read_image(batch_size, batch_times):
            start_time = time.time()
            _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
            # 统计平均结果
            train_loss += err
            train_acc += ac
            n_batch += 1
            end_time = time.time()
            diff = end_time - start_time
            print("epoch: %d, n_batch: %d, err: %f, ac: %f, diff: %.2f" % (epoch, n_batch, err, ac, diff))
            sys.stdout.flush()
            if ac > max_acc:
                max_acc = ac
                saver.save(sess, table + '/pic-cat')
            del x_train_a
            del y_train_a
            # print("   train loss: %f" % (train_loss / n_batch))
            # print("   train acc: %f" % (train_acc / n_batch))
            # sys.stdout.flush()
    sess.close()


# 检测数据
def check(x, y_, logits):
    sess = tf.InteractiveSession()

    saver = tf.train.Saver(max_to_keep=1)
    # 读取保存的模型
    model_file = tf.train.latest_checkpoint(table+'/')
    if model_file is None:
        # 如果没有保存模型，初始化所有变量
        sess.run(tf.global_variables_initializer())
    else:
        # 如果保存了模型，直接从模型中初始化变量
        saver.restore(sess, model_file)

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
    query_sql = "select count(*) from %s" % table
    # 执行sql语句
    cursor.execute(query_sql)
    # 获取查询结果
    result = cursor.fetchone()
    # 在所有图片中随机取一张图片
    id = random.randint(1, result[0])
    query_sql = "select * from %s where id >= %d limit 1" % (table, id)
    # 执行sql语句
    cursor.execute(query_sql)
    # 获取查询结果
    result = cursor.fetchone()
    db.close()
    try:
        # get请求图片url
        url = result[2][:result[2].index('?')]
        response = requests.get(url)
    except:
        print('unable to open pic')
        return
    # 图片请求能正常响应
    imgs = []
    lables = []
    if response.status_code == 200:
        # 获取网络图片
        try:
            # 加载图片
            img = io.imread(BytesIO(response.content))
            arr = transform.resize(img, (width, height))
            del img
            # print(arr.shape, arr.dtype)
        except:
            print('unable to open pic')
            return
        # 填充list
        if len(arr.shape) == 3 and arr.shape[2] == channel:
            imgs.append(arr)
            lables.append(cat_arr.index(result[5]))
        del arr

    x_train_a = np.asarray(imgs, np.float32)
    y_train_a = np.asarray(lables, np.int32)
    del imgs
    del lables

    logit = sess.run(logits, feed_dict={x: x_train_a, y_: y_train_a})
    # 图片id
    print("id : %d, url : %s" % (result[0], result[1]))
    # 正确分类
    print("correct : ", result[2])
    # 预测分类
    print("forecast : ", cat_arr[np.argmax(logit)])
    sess.close()


if __name__ == '__main__':
    arg_len = len(sys.argv)
    if arg_len <= 1 or sys.argv[1] == 'train':
        # 读取分类
        cat_arr = read_cat()
        x, y_, train_op, loss, acc, logits = model()
        n_epoch = 10 if arg_len < 3 else int(sys.argv[2])
        batch_size = 10 if arg_len < 4 else int(sys.argv[3])
        batch_times = 100 if arg_len < 5 else int(sys.argv[4])
        train(x, y_, train_op, loss, acc, n_epoch, batch_size, batch_times)
    elif sys.argv[1] == 'check':
        # 读取分类
        cat_arr = read_cat()
        x, y_, train_op, loss, acc, logits = model()
        check(x, y_, logits)
    else:
        print('param invalid')
