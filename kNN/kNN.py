# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 09:51:34 2017

@author: duanzhiying
"""
import numpy as np
import pandas as pd
import operator

"""创建数据集
"""
def createDataset():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

"""最简单的KNN分类算法
参数：
    inX：待分类的测试变量
    dataSet：训练数据集
    labels：训练数据集的分类/标签信息
    k：设定的K值
"""
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] # 获取数据集的条数（shape返回一个元组：（行，列））
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet # 相减（tile方法用于在行/列上重复）
    sqDiffMat = diffMat**2 # 平方
    sqDistances = sqDiffMat.sum(axis=1) # 求和（axis＝0表示按列方向，axis＝1表示按行方向，结果为1行dataSetSize列）
    distances = sqDistances**0.5 # 开平方
    sortedDistIndicies = distances.argsort() # 对distances进行升序排序（距离最近的），返回序号下标  
    classCount={} # 保存每个分类的数量          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]] 
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) # 按照classCount的第二个值即value进行排序
    return sortedClassCount[0][0] # 返回value最大的分类

"""读取文件转换为矩阵
   返回参数：
   returnMat：特征矩阵
   classLabelVector：标记列表
"""
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = np.zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def file2matrixNew(filename):
    data =pd.read_table(filename,sep='\t',header=None)
    returnMat = data[list(range(3))]
    classLabelVector = np.array(data[[3]]).tolist()
    return returnMat,classLabelVector
    