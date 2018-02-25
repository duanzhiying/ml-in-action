#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 14:05:57 2018

@author: tracyduan
"""
import numpy as np
import svd.svd as svd


#myMat = np.mat(svd.loadExData())
#myMat[0,1] = myMat[0,0] = myMat[1,0] = myMat[2,0] = 4
#myMat[3,3] = 2
#
#result = svd.recommend(myMat, 2)

svdMat = np.mat(svd.loadExData2())
result2 = svd.recommend(svdMat, 1, estMethod = svd.svdEst)
svd.imgCompress(2)