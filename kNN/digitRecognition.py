# -*- coding: utf-8 -*-
import numpy as np
from os import listdir

def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        linestr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(linestr[j])
    return returnVect

'''
    dir：保存训练文本或者测试文本的文件夹名称
'''
def dataSetBuild(dir):
    fileList = listdir(dir)
    hwLabel = []
    m = len(fileList)
    dataMat = np.zeros((m,1024))

    for i in range(m):
        fileNameStr = fileList[i]
        classNumStr = int(fileNameStr.split('_')[0])
        hwLabel.append(classNumStr)
        dataMat[i,:] = img2vector(dir + '/' + fileNameStr)
    return m, dataMat, hwLabel
        
        
    

    