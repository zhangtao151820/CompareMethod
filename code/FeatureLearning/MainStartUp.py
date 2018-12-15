# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 14:52:27 2018
@author: zhang
"""
from keras.models import load_model
import numpy as np
import scipy.io as sio
import allImageDataPath as ALLPATH
import AllCNNModelFunction as modelCNN
import ReadImageFromPath as prepData
import ModelCompileFitPredict as ModelCFP
import datetime

finalclass = 2;    ExtentNum =2;
# 原始影像的路径，读取到8个波段的数据，上采样至15米
PanPath = ALLPATH.BJpanP;    BJmultyBands = ALLPATH.BJMbandsP;   shapePath = ALLPATH.BJSample;  
ImageData = prepData.ReadImageBands(PanPath,BJmultyBands);
Rows = ImageData.shape[0];    Cols = ImageData.shape[1];     numberBands =  ImageData.shape[2];


BJ3PCA = ALLPATH.BJ3PCA;    BJsharpenRGB = ALLPATH.BJsharpenRGB; 
PCAData = prepData.ReadPCA3orRGB(BJ3PCA);
RGBData = prepData.ReadPCA3orRGB(BJsharpenRGB);


starttime = datetime.datetime.now();               
IsImagenet = True;     IsTrain = False;
# 运用自定义的CNN结构 或者 基于Imagenet训练的网络
if (IsImagenet == False):
    WS = 2*ExtentNum+1;
    # 获取样本点的行列号以及Label
    RowsCols,allLabel = prepData.GetXYValueToRowsCols(PanPath,shapePath);
    SamplePathes,finalUseLabel = prepData.MakeSample(RowsCols,allLabel,IsImagenet,ImageData,ExtentNum,1);
    # 训练集和测试集的划分（分层随机划分，测试集比例0.4） 
    X_train,X_test,y_trainfinal,y_testfinal = prepData.SplitSample(SamplePathes,finalUseLabel,0.4,finalclass); 
    if (IsTrain == True):
         # 调用自己搭建的的-CNN模型，用训练集进行训练和拟合
        CNNmodel = ModelCFP.ModelcompileFit( modelCNN.Conv2DCNN(WS,numberBands,finalclass), 
                                              X_train, y_trainfinal,0.4  );
        CNNmodel.save(r'F:\L8PanBuild\PYCODE\CNNbuilt\CNNBJModel\BJCNNConv2D55.h5');
    else:
        LoadCNNmodel = load_model(r'F:\L8PanBuild\PYCODE\CNNbuilt\CNNBJModel\BJCNNConv2D55.h5');
        # 利用测试集进行精度评价，并利用模型对所有数据进行分类
        PredictLabel,score,lenthTest,Conv2DCNNHHmat = ModelCFP.PredictALLdata(ImageData,ExtentNum,
                                                      X_test,y_testfinal,LoadCNNmodel,Rows,Cols,1);
        prepData.Draw2orMoreClassResult(PredictLabel,'CNNBJ');
        sio.savemat(r'F:\L8PanBuild\PYCODE\CNNbuilt\CNNBJModel\BJMYCNN2DLabel55.mat',{'PL':PredictLabel});

else:
    #主成分分析，将8个波段降维至3个波段
    tempLabel = [];      WS = 50;
    # 获取样本点的行列号以及Label
    RowsCols,allLabel = prepData.GetXYValueToRowsCols(PanPath,shapePath);
    # PCA 3bands
    SamplePathesPCA,finalUseLabelPCA = prepData.MakeSample(RowsCols,allLabel,True,PCAData,ExtentNum,10);
    # 训练集和测试集的划分（分层随机划分，测试集比例0.4） 
    X_trainPCA,X_testPCA,y_trainPCA,y_testPCA = prepData.SplitSample(SamplePathesPCA,finalUseLabelPCA,0.4,finalclass);
    # RGB
    SamplePathesRGB,finalUseLabelRGB = prepData.MakeSample(RowsCols,allLabel,True,RGBData,ExtentNum,10);
    X_trainRGB,X_testRGB,y_trainRGB,y_testRGB = prepData.SplitSample(SamplePathesRGB,finalUseLabelRGB,0.4,finalclass);
    
    
    if (IsTrain == True):
        # 调用微调的IamgeNet-CNN模型，用训练集进行训练和拟合
        vgg16modelPCA = ModelCFP.ModelcompileFit( modelCNN.VGG16Finetuning(WS), X_trainPCA, y_trainPCA,0.4);
        vgg16modelPCA.save( r'F:\L8PanBuild\PYCODE\CNNbuilt\CNNBJModel\vgg16modelPCA.h5' );
        
        vgg16modelRGB = ModelCFP.ModelcompileFit( modelCNN.VGG16Finetuning(WS), X_trainRGB, y_trainRGB,0.4);
        vgg16modelRGB.save( r'F:\L8PanBuild\PYCODE\CNNbuilt\CNNBJModel\vgg16modelRGB.h5' );
        
    else:
        LoadVGG16modelPCA = load_model(r'F:\L8PanBuild\PYCODE\CNNbuilt\CNNBJModel\vgg16modelPCA.h5');
        # 利用测试集进行精度评价，并利用模型对所有数据进行分类
        PredictLabelPCA,scorePCA,lenthTestPCA,Conv2DCNNHHmatPCA = ModelCFP.PredictALLdata(PCAData,ExtentNum,
                                                      X_testPCA,y_testPCA,LoadVGG16modelPCA,Rows,Cols,10);
        prepData.Draw2orMoreClassResult(PredictLabelPCA,'VGG16BJPCA');
        sio.savemat(r'F:\L8PanBuild\PYCODE\CNNbuilt\CNNBJModel\BJVGG162DPCA55.mat',{'PL':PredictLabelPCA});
                                                      
        LoadVGG16modelRGB = load_model(r'F:\L8PanBuild\PYCODE\CNNbuilt\CNNBJModel\vgg16modelRGB.h5');
        # 利用测试集进行精度评价，并利用模型对所有数据进行分类
        PredictLabelRGB,scoreRGB,lenthTestRGB,Conv2DCNNHHmatRGB = ModelCFP.PredictALLdata(RGBData,ExtentNum,
                                                      X_testRGB,y_testRGB,LoadVGG16modelRGB,Rows,Cols,10);
        prepData.Draw2orMoreClassResult(PredictLabelRGB,'VGG16BJRGB');
        sio.savemat(r'F:\L8PanBuild\PYCODE\CNNbuilt\CNNBJModel\BJVGG162DRGB55.mat',{'PL':PredictLabelRGB});
        
endtime = datetime.datetime.now();
print ( (endtime - starttime).seconds );

