# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 16:52:37 2018

@author: duanzhiying
"""
import numpy as np

def loadSimpData():
    datMat = np.matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

"""对数据进行分类，默认都是1（即一个分类结果）
   对于lt类型，将小于等于threshVal的置为-1，对于gt类型，将大于threshVal的置为-1。
   为什么区分lt和gt是对分类结果的两种情况进行尝试，因为而分类的yes or no在左边还是右边未知
"""
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    retArray = np.ones((np.shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

"""输入参数：数据集，分类列表，权重向量
"""
def buildStump(dataArr,classLabels,D):
    # 获取data和label的矩阵。其中label转置为纵向
    dataMatrix = np.mat(dataArr); labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)
    # bestClasEst为m行1列的0数据
    numSteps = 10.0; bestStump = {}; bestClasEst = np.mat(np.zeros((m,1)))
    # 最小误差，默认为无穷大
    minError = np.inf 
    # 遍历所有特征维度
    for i in range(n):
        # 得到该特征的最大和最小值
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        # 存在步长的概念
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
            for inequal in ['lt', 'gt']: #go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
                errArr = np.mat(np.ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr  #calc total error multiplied by D
                #print ("split: dim %d, thresh %.2f, predictedVals: %s thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, str(predictedVals.T), inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

"""基于单层决策树（Decision Stump）的AdaBoosting训练过程
   输入参数：numIt是迭代次数，需要用户指定
"""
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    # 弱分类器
    weakClassArr = []
    m = np.shape(dataArr)[0]
    # D是概率分布向量，初始化均分
    D = np.mat(np.ones((m,1))/m)   
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        # 构造单层决策树
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        #print ("D:",D.T)
        # 计算alpha值，alpha值也是第一次分类的系数；1e-16防止分母error为0
        alpha = float(0.5 * np.log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha 
        # 保存迭代的最好的分类信息
        weakClassArr.append(bestStump)                  
        #print ("classEst: ",classEst.T)
        # 计算新权值分布D的e指数
        expon = np.multiply(-1*alpha* np.mat(classLabels).T,classEst) 
        # 如下两步计算新D
        D = np.multiply(D, np.exp(expon))                              
        D = D/D.sum()
        # 对分类结果作用alpha系数（权重）
        aggClassEst += alpha*classEst
        #print ("aggClassEst: ",aggClassEst.T)
        # 计算分类的错误率。numpy的sign(x)函数：x大于0的返回1,小于0的返回-1,等于0的返回0
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T,np.ones((m,1)))
        errorRate = aggErrors.sum()/m
        print ("total error: ",errorRate)
        # 如果错误率为0，则退出循环
        if errorRate == 0.0: break
    return weakClassArr,aggClassEst

"""AdaBoosting分类函数
"""
def adaClassify(datToClass, classifierArr):
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        # 分类结果将随着迭代越来越强
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print (aggClassEst)
    return np.sign(aggClassEst)
        
    
datMat, classLabels = loadSimpData()
#D = np.mat(np.ones((5,1))/5)
#bestStump, minError, bestClasEst = buildStump(datMat,classLabels,D)
classfifierArray, aggClassEst = adaBoostTrainDS(datMat, classLabels, 30)
cls = adaClassify(datMat, classfifierArray)