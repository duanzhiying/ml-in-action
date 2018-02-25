# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:30:31 2018

@author: duanzhiying
"""
import numpy as np
import regression.lr as lr

#xArr,yArr = lr.loadDataSet('../data/ex0.txt')
#ws = lr.standRegres(xArr, yArr)
#
#yHat = lr.lwlrTest(xArr, xArr, yArr, 0.003)
#
#xMat = np.mat(xArr)
#srtInd = xMat[:,1].argsort(0)
#xSort = xMat[srtInd][:,0,:]
#
#import matplotlib.pyplot as plt
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot(xSort[:,1],yHat[srtInd])
#ax.scatter(xMat[:,1].flatten().A[0], np.mat(yArr).T.flatten().A[0], s=2, c='red')
#plt.show()
        
abX,abY = lr.loadDataSet('../data/abalone.txt')
#idgeWeights = lr.ridgeTest(abX,abY)
lr.stageWise(abX, abY, 0.01, 200)

#import matplotlib.pyplot as plt
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot(ridgeWeights)
#ax.set_title('log(lambda)')
#plt.show()