# -*- coding: utf-8 -*-

from kNN import kNN
import matplotlib.pyplot as plt
from numpy import array

group,label = kNN.createDataset()
targetLabel = kNN.classify0([1,1],group,label,3)

datingDataMat, datingLabels = kNN.file2matrixNew('../data/datingTestSet2.txt')

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
#plt.show()
# c 是color，maker是尺寸。即利用15 *　datingLabels的1， 2， 3作为不同点的颜色和尺寸。
#ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
## 更适合用jupyter来所见即所得地呈现图表形状
#plt.show()
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels),15.0*array(datingLabels))
# 更适合用jupyter来所见即所得地呈现图表形状
plt.show()


