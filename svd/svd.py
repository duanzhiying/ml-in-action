#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 13:59:00 2018

@author: tracyduan
"""

from numpy import *
from numpy import linalg as la

def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]
    
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0]]
    
def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))

def pearsSim(inA,inB):
    if len(inA) < 3 : return 1.0
    return 0.5+0.5*corrcoef(inA, inB, rowvar = 0)[0][1]

def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

def standEst(dataMat, user, simMeas, item):
    # 得到商品总数
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0: continue
        # 取item和第j个商品都有评分的user下标
        overLap = nonzero(logical_and(dataMat[:,item].A>0, \
                                      dataMat[:,j].A>0))[0]
        if len(overLap) == 0: 
            similarity = 0
        else: 
            # 计算相似度：取不为0的值
            similarity = simMeas(dataMat[overLap,item], \
                                   dataMat[overLap,j])
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        # 保存总的相似度
        simTotal += similarity
        # 评分：相似度 * 评分
        ratSimTotal += similarity * userRating
    if simTotal == 0: 
        return 0
    else: 
        # 返回平均评分
        return ratSimTotal/simTotal
    
"""基于SVD的相似度估计
"""
def svdEst(dataMat, user, simMeas, item):
    # 获取item总数
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    U,Sigma,VT = la.svd(dataMat)
    # 取前4个Sigma组成对角矩阵，4行4列
    Sig4 = mat(eye(4)*Sigma[:4])
    # 生成转化后的矩阵：11行4列，问题为什么这里没有VT
    xformedItems = dataMat.T * U[:,:4] * Sig4.I
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item: continue
        similarity = simMeas(xformedItems[item,:].T,\
                             xformedItems[j,:].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: 
        return 0
    else: 
        return ratSimTotal/simTotal

def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    # 获取该user未评分的item下标（列下标）
    unratedItems = nonzero(dataMat[user,:].A==0)[1] 
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]

"""打印矩阵
"""
def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                # python3默认是换行的
                print(1,end=' '),
            else: print(0,end=' '),
        print('')

"""图像压缩
"""
def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('../data/0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print("****original matrix******")
    printMat(myMat, thresh)
    U,Sigma,VT = la.svd(myMat)
    SigRecon = mat(zeros((numSV, numSV)))
    # 根据Sigma构造对称矩阵，仅取前numSV个奇异值
    for k in range(numSV):
        SigRecon[k,k] = Sigma[k]
    # 重新构造矩阵，即截断U和V
    reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:]
    print("****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh)
