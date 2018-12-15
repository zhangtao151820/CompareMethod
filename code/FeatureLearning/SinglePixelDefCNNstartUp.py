# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 10:12:35 2018

@author: zhang
"""

from keras.models import load_model
import scipy.io as sio
import allImageDataPath as ALLPATH
import AllCNNModelFunction as modelCNN
import ReadImageFromPath as prepData
import ModelCompileFitPredict as ModelCFP


finalclass = 2;    
# 原始影像的路径，读取到8个波段的数据，上采样至15米
PanPath = ALLPATH.BJpanP;     multyBands = ALLPATH.BJMbandsP;  shapePath = ALLPATH.BJSample;  
ImageData = prepData.ReadImageBands(PanPath,multyBands);
Rows = ImageData.shape[0];    Cols = ImageData.shape[1];     numberBands =  ImageData.shape[2];


IsTrain = False;
# 获取样本点的行列号以及Label
RowsCols,allLabel = prepData.GetXYValueToRowsCols(PanPath,shapePath);
SamplePathes,finalUseLabel = prepData.MakeOnePixelSamples(RowsCols,allLabel,ImageData);
# 训练集和测试集的划分（分层随机划分，测试集比例0.4） 
X_train,X_test,y_trainfinal,y_testfinal = prepData.SplitSample(SamplePathes,finalUseLabel,0.4,finalclass); 
if (IsTrain == True):
    # 调用自己搭建的的-CNN模型，用训练集进行训练和拟合
    CNNmodel = ModelCFP.ModelcompileFit( modelCNN.Conv1DdefCNN(numberBands,finalclass), 
                                              X_train, y_trainfinal,0.4);
    CNNmodel.save(r'F:\L8PanBuild\PYCODE\CNNbuilt\CNNBJModel\BJOnePixelConv1D.h5');
else:
    LoadCNNmodel = load_model(r'F:\L8PanBuild\PYCODE\CNNbuilt\CNNBJModel\BJOnePixelConv1D.h5');
    # 利用测试集进行精度评价，并利用模型对所有数据进行分类
    PredictLabel,score,lenthTest,Conv2DCNNHHmat = ModelCFP.PredictLabelConv1D( ImageData,
                                                     X_test,y_testfinal,LoadCNNmodel,Rows,Cols);
    prepData.Draw2orMoreClassResult(PredictLabel,'CNNBJ');
    sio.savemat(r'F:\L8PanBuild\PYCODE\CNNbuilt\CNNBJModel\BJMYPixelConv1D.mat',{'PL':PredictLabel});
    
    
