# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 10:54:13 2018

@author: duanzhiying
"""

import pca.pca as pca
import matplotlib.pyplot as plt

#data = pca.loadDataSet('../data/testSet_pca.txt')
#lowDMat, reconMat = pca.pca(data,1)
#
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.scatter(data[:,0].flatten().A[0],data[:,1].flatten().A[0],
#           marker = '^', s=90)
#ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],
#           marker = 'o', s=50, c='red')


# 如下测试大规模数据集：半导体
data = pca.loadDataSet('../data/secom.data',' ')
datMat = pca.replaceNanWithMean(data)
