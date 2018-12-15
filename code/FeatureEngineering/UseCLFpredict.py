# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 15:46:07 2018

@author: zhang
"""
from sklearn.externals import joblib
import scipy.io as sio
import PrepareSampleXData as PPSD
import OptimalSuperParameter as OSP
import TrainClassiferModel as TCFP

# 加载模型
BestclfRFC = joblib.load( r'F:\L8PanBuild\BIGREGION\saveCLF\BJSVMBands33.model' );

# 对整个区域进行分类
ResultLabel = OSP.UseModelToPredict(BestclfRFC, TCFP.AllFatureSelct(0), 1);  # 0表示不扩充，不考虑空间邻域
# 分类结果绘制
PPSD.Draw2orMoreClassResult(ResultLabel,'RFresult');      # AdaBoost
sio.savemat(r'F:\L8PanBuild\BIGREGION\ClassOutput\MAT\BJSVMBands33.mat',{'FL':ResultLabel});