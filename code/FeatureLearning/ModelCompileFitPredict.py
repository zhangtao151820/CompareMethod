
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 11:15:20 2018

@author: zhang
"""
import math,numpy
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD,Adam
import DrawLossAndAccChart as DLAC
import ReadImageFromPath as prepData
import scipy.ndimage as SNDIG
from sklearn.metrics import confusion_matrix

#----多光谱数据（8个波段）最近邻上采样，分辨率从30变为15米---------------------------#
def MuBandsUp(inputsample,scaleX,scaleY):
    outputSample=[];  number=inputsample.shape[2];
    tempdstack = SNDIG.zoom(inputsample[:,:,0], (scaleX,scaleY), order=0);
    for ii in numpy.arange(1,number):
        tempdstack = numpy.dstack((tempdstack,SNDIG.zoom(inputsample[:,:,ii], (scaleX,scaleY), order=0) ));
    outputSample = tempdstack;
    return numpy.array(outputSample);
#-----------------------------------------------------------------------------------#

#------------------模型的编译和训练，返回训练好的模型---------------------------------#
def ModelcompileFit(model,x_train, y_train,Val_Rate):
    def step_decay(epoch):
        initial_lrate = 0.0001
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
        if lrate<=0.000001:
            lrate = 0.000001
        return lrate;
    mylrate = LearningRateScheduler(step_decay);
    #sgd = SGD(lr = 0.001, momentum = 0.9, decay=0.0, nesterov=False);
    sgd = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer = sgd,metrics=['accuracy']);
    #创建一个实例history
    history = DLAC.LossHistory();
    model.fit(x_train, y_train, batch_size=256, epochs=200,verbose=1, 
               validation_split=Val_Rate,callbacks=[history,mylrate] );
    history.loss_plotMytime('epoch');
    return model;
#-----------------------------------------------------------------------------------#    
#多类别分类最终类别判定
def finalLable2orMoreClass(predictY):
    finalY= 0*numpy.arange(0,predictY.shape[0]);
    for jj in range(predictY.shape[0]):
        temp = predictY[jj];
        index = numpy.where( temp == numpy.max(temp) )
        indexArray = list(index) 
        finalY[jj] = indexArray[0][0] + 1;
    return finalY;    
#------------------------------------------------------------------------------------#    
#-----利用训练好的CNN对所有待分类数据预测Label----图像patch------测试集的精度-----------#
def PredictALLdata(allImageData,EXT,testData,testLabel,Trainedmodel,Rows,Cols,scale):
    # Nubands = allImageData.shape[2];
    afterExtent = prepData.MubandsWindowSizeEX(allImageData,EXT);  #原始多波段数据外围进行扩展
    #--------------------对所有数据进行分类-----------#
    PredictLabel = numpy.zeros((Rows, Cols),dtype = numpy.int8);    # WS = 2*EXT+1;
    for i in numpy.arange(0,Rows):
        Hindex = i+EXT;
        HangData = afterExtent[Hindex-EXT:Hindex+EXT+1,:,:];    #每次对原始数据的一行进行分类预测
        #HangPathes = numpy.zeros((Cols,WS, WS,Nubands),dtype = numpy.uint16);
        HangPathes = [];
        for kk in numpy.arange(0,Cols):
            Lind = kk + EXT;
            onepath =  HangData[:,Lind-EXT:Lind+EXT+1,:];
            if scale <= 1:
                HangPathes.append(  numpy.array(onepath) );
            else:
                HangPathes.append(  MuBandsUp(onepath,scale,scale) );
        HangpredictY = Trainedmodel.predict( numpy.array(HangPathes), batch_size=256, verbose=0);
        HangfinalY = finalLable2orMoreClass(HangpredictY);
        PredictLabel[i,:] = numpy.int8(HangfinalY); 
    #------------------------------------------------#  
    #------------------测试集的精度--------------------#
    lenthTest = testLabel.shape[0];            #测试样本的样本数量  
    #测试样本的预测结果Label
    testPredict =  finalLable2orMoreClass( Trainedmodel.predict( testData, batch_size=256, verbose=0) );
    HHmat = confusion_matrix( finalLable2orMoreClass(testLabel), testPredict);    #混淆矩阵
    score = Trainedmodel.evaluate(testData, testLabel, batch_size=256,verbose=0);        # 测试集平均精度
    #-------------------------------------------------#
    return PredictLabel, score, lenthTest, HHmat;    

#-----利用训练好的CNN对所有待分类数据预测Label---单像素-------测试集的精度-----------------#
def PredictLabelConv1D(allImageData,testData,testLabel,Trainedmodel,Rows,Cols):
    #--------------------对所有数据进行分类-----------#
    PredictLabel = numpy.zeros((Rows, Cols),dtype = numpy.int8);
    for i in numpy.arange(0,Rows):
        HangData = allImageData[i,:,:];    #每次对原始数据的一行进行分类预测
        HangPathes = [];     BUb = allImageData.shape[2];
        for kk in numpy.arange(0,Cols):
            onepath =  HangData[kk,:];
            HangPathes.append(  onepath.reshape(BUb,1)  );
        HangpredictY = Trainedmodel.predict( numpy.array(HangPathes), batch_size=256, verbose=0);
        HangfinalY = finalLable2orMoreClass(HangpredictY);
        PredictLabel[i,:] = numpy.int8(HangfinalY); 
    #------------------------------------------------# 
    #------------------测试集的精度--------------------#
    lenthTest = testLabel.shape[0];            #测试样本的样本数量  
    #测试样本的预测结果Label
    testPredict =  finalLable2orMoreClass( Trainedmodel.predict( testData, batch_size=256, verbose=0) );
    HHmat = confusion_matrix( finalLable2orMoreClass(testLabel), testPredict);    #混淆矩阵
    score = Trainedmodel.evaluate(testData, testLabel, batch_size=256,verbose=0);        # 测试集平均精度
    #-------------------------------------------------#
    return PredictLabel, score, lenthTest, HHmat;  
