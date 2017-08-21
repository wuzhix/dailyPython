# -*- coding: UTF-8 -*-

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 读取图片
im = Image.open('./../resource/1.jpg')
# 显示图片
im.show()
# 转换为数组
arr = np.array(im)
# 输出图片信息
print(arr.shape, arr.dtype)
# 显示图片
plt.imshow(arr)
# 不显示坐标轴
plt.axis('off')
plt.show()
# 红色通道
r = arr[:, :, 0]
# 交换红蓝通道并显示
arr[:, :, 0] = arr[:, :, 2]
arr[:, :, 2] = r
# 显示图片
plt.imshow(arr)
# 不显示坐标轴
plt.axis('off')
plt.show()
im = Image.fromarray(arr)
im.show()
