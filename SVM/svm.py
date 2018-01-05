# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:03:10 2018

@author: duanzhiying
"""
    
import pandas as pd
import numpy as np

def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(np.random.uniform(0,m))
    return j

# 对alphaj进行剪辑，考虑L和H
def clipAlpha(aj,H,L):
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj
    
def smoSimple(dataArr, labelArr, C, toler, maxIter):
    dataMatrix = np.asmatrix(dataArr)
    labelMat = np.asmatrix(labelArr)
    m,n = np.shape(dataMatrix)
    iter = 0
    b = 0
    alphas = np.mat(np.zeros((m,1))) # m行1列的矩阵
    while (iter < maxIter):
        alphaPairsChanged = 0 # 记录alpha是否已经进行了优化
        for i in range(m):
            # multiply是矩阵的数量积。fXi对应7.104 g(x)
            fXi = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            # Ei对应7.105：为函数g(x)对输入xi的预测值与真实输出yi之差
            Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions
            # 如果误差很大就对该数据相应的alpha做优化，正负间隔都会被优化，并检查alpha值在0和C之间
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                # 确保选择的j不和i相同
                j = selectJrand(i,m)
                fXj = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
                # label的符号不一致：y1!=y2
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else: # y1=y2
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print("L==H"); continue
                # 对应于7.107，是计算alphaj的中间变量
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print("eta>=0"); continue
                # 对应于式7.106，沿着约束方向未经剪辑时的解
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                # 按照L和H进行剪辑
                alphas[j] = clipAlpha(alphas[j],H,L)
                # 若该alpha值不再变化，则停止对它的优化
                if (abs(alphas[j] - alphaJold) < 0.00001): 
                    print("j not moving enough"); 
                    continue
                # 对应于式7.109，由alphaJ求得alphaI
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
                # 对应于式7.115 设置常数项                                                        #the update is in the oppostie direction
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                # 对应于式7.116 
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): 
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): 
                    b = b2
                else: 
                    b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            # 如果alpha不再改变则迭代次数就加1
            if (alphaPairsChanged == 0): 
                iter += 1
            else:
                iter = 0
            print("iteration number: %d" % iter)
    return b,alphas
    
"""创建数据集
   返回：array
"""
def createDataset(fileName):
    df = pd.read_table(fileName,header=None)
    return df[[0,1]].values,df[[2]].values

"""计算w值，对应式7.25
"""  
def calcWs(alphas,dataArr,classLabels):
    X = np.asmatrix(dataArr)
    labelMat = np.asmatrix(labelArr)
    m,n = np.shape(X)
    w = np.zeros((n,1))
    for i in range(m):
        w += np.multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

# 如下是测试运行代码
dataArr, labelArr = createDataset("../data/testSet.txt")
b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
ws = calcWs(alphas,dataArr,labelArr)
