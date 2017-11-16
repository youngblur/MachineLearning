#encoding = utf-8

'''
    使用Matplotlib 创建散点图
'''

import matplotlib
import matplotlib.pyplot as plt
from Classification.KNN import kNN
from numpy import array

datingDataMat = kNN.datingDataMat
datingLabels = kNN.datingLabels
fig  = plt.figure()
ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:,1],datingDataMat[:,2])       #以 第二列为 横坐标 第三列为 纵坐标
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()