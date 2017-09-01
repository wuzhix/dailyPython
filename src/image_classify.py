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

图像匹配系统流程：
分类训练系统
1、批量读入图像，通过卷积，池化得到抽象特征
2、通过交叉熵损失函数，Adam优化算法计算权重矩阵
3、抽象特征*权重矩阵的结果，经过softmax归一化，得出图像的分类概率
4、每张图片概率最大的分类与正确分类进行比较，计算准确率
5、保存训练结果
匹配系统
1、保存每张图片的抽象特征*权重矩阵的结果
2、每种分类选择一张图片作为基础图片，该分类的其他图片计算与该图片的欧氏距离
3、设置相似图片欧氏距离为想<=n
4、任选一张图片，通过训练结果得出该图片分类，找到该分类的基础图片
5、计算该图片与基础图片的欧氏距离m
6、从该分类中查出所有与基础图片欧氏距离在m-n到m+n之间的图片，如果m-n<=则范围是0到m+n之间
7、重新计算该图片与图片集中的图片的欧氏距离，过滤掉欧氏距离>n的图片
8、剩下的图片即为相似图片
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

# 获取分类
cat_arr = []
start_id = 1
# 每次查询条数
batch_size = 100
# 每一轮循环批量训练的次数
batch_times = 20000


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
        query_sql = "select distinct(two) from %s where two != 0" % table
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
def read_image():
    global start_id
    # 连接数据库，这里一定要加上charset，不然读取的中文会乱码
    db = pymysql.connect(host=server, port=db_port, user=username, password=password, database=database, charset="utf8")
    cur_times = 0
    while True:
        db.ping()
        # 使用cursor()方法获取操作游标
        cursor = db.cursor()
        # 查询语句
        query_sql = "select * from %s where id >= %d and three != 0 limit %d" % (table, start_id, batch_size)
        # 执行sql语句
        cursor.execute(query_sql)
        # 获取查询结果
        result = cursor.fetchall()
        start_id += len(result)
        imgs = []
        lables = []
        # print(len(result))
        for i, row in enumerate(result):
            # requests.get(url)和Image.open(BytesIO(response.content))都会抛出异常
            try:
                # get请求图片url
                pos = row[2].find('?')
                if pos != -1:
                    row[2] = row[2][:pos]
                response = requests.get(row[2])
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
                    lables.append(cat_arr.index(row[4]))
                del arr
            del response
        # asarray将list转换为ndarray
        yield np.asarray(imgs, np.float32), np.asarray(lables, np.int32)
        del imgs
        del lables
        cur_times += 1
        if cur_times >= batch_times:
            break


# 获取分类
cat_arr = read_cat()


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
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
# 结果是否匹配
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
# 计算准确率
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 训练次数
n_epoch = 10
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
    for x_train_a, y_train_a in read_image():
        if len(x_train_a) > 0 and len(y_train_a) > 0:
            start_time = time.time()
            _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
            # 统计平均结果
            train_loss += err
            train_acc += ac
            n_batch += 1
            end_time = time.time()
            diff = end_time - start_time
            cur_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print("%s epoch: %d, n_batch: %d, err: %f, ac: %f, time: %.2f" % (cur_date, epoch, n_batch, err, ac, diff))
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
