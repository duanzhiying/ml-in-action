# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:13:39 2018

@author: duanzhiying
"""

from adaBoosting import adaBoosting
import pandas as pd
import numpy as np

def loadDataSet(fileName):
    data = pd.read_table(fileName, header = None)
    length = data.shape[1]
    return data.iloc[:,:length-1], data.iloc[:,length-1]

feature, label = loadDataSet("../data/horseColicTraining2.txt")
classifierArray, aggClassEst = adaBoosting.adaBoostTrainDS(feature, label, 10)

# 预测，错误率统计
testFeature, testLabel = loadDataSet("../data/horseColicTest2.txt")
prediction10 = adaBoosting.adaClassify(testFeature, classifierArray)

len = testFeature.shape[0]
errArr = np.mat(np.ones((len,1)))
errArr[prediction10 != np.mat(testLabel).T].sum()




    