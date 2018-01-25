# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 10:52:10 2018

@author: duanzhiying
"""
import numpy as np

"""载入数据集
"""
def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    # python2的版本没有list包装，得到的mat是1行1000列的，正常的应该是1000行2列，需要研究为什么这样
    datArr = [list(map(float,line)) for line in stringArr]
    return np.mat(datArr)

"""pca主成分分析程序
   输入参数：dataMat数据集（矩阵形式），topNfeat所取的特征数量
"""
def pca(dataMat, topNfeat=9999999):
    # 平均值。axis=1的话是按行处理，得到的是1000行1列，axis=0是按列，得到的是1行2列
    meanVals = np.mean(dataMat, axis=0)
    # 减去均值
    meanRemoved = dataMat - meanVals
    # 协方差，特征按列，2个特征，协方差矩阵式2行2列
    covMat = np.cov(meanRemoved, rowvar=0)
    # 计算协方差矩阵的特征值和特征向量
    eigVals,eigVects = np.linalg.eig(np.mat(covMat))
    # 对特征值进行排序，升序
    eigValInd = np.argsort(eigVals)   
    # 倒序（特征值从大到小）取topNfeat个特征值（需要看下花式切片的技巧）      
    eigValInd = eigValInd[:-(topNfeat+1):-1] 
    # 取特征值对应的特征矩阵
    redEigVects = eigVects[:,eigValInd]       
    # 将数据转换到N个特征向量构建的新空间
    lowDDataMat = meanRemoved * redEigVects
    # 按照逆转换 到原来的 空间数据，方便作图查看
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat

"""将NaN替换成平均值
"""
def replaceNanWithMean(datMat): 
    numFeat = np.shape(datMat)[1]
    for i in range(numFeat):
        meanVal =np.mean(datMat[np.nonzero(~np.isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
        datMat[np.nonzero(np.isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat
