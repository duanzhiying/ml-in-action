# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 18:52:41 2018

@author: duanzhiying
"""
import numpy as np

"""创建数据集
"""
def createDataset():
    group = np.array([[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']])
    labels = ['no surfacing', 'flippers']
    return group, labels

#def splitDataSet(dataSet, axis, value):
#    retDataSet = []
#    for featureVect in dataSet:
#        if featureVect[axis] == value: