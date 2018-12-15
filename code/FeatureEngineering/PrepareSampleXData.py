# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 09:20:34 2018
#训练样本和测试数据
@author: zhang
"""

import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#对于单波段，外围扩充一圈
def ExtentOneCircle(A):
    Rows=A.shape[0];         #Cols=A.shape[1];
    LRE= np.concatenate((A[:,0].reshape(Rows,1),A, A[:,-1].reshape(Rows,1)),axis=1);
    COLSLRE=LRE.shape[1];    # ROWSLRE=LRE.shape[0];   
    UpDownExtent=np.concatenate((LRE[0,:].reshape(1,COLSLRE),LRE,LRE[-1,:].reshape(1,COLSLRE)),axis=0);
    return UpDownExtent;

#对于单波段，外围扩充1，2，3，4, 5, 6圈
def WindowSizeFunction(A,extentnumber):
    if(extentnumber>=5):
        print("外围扩展次数必须小于5");
    elif(extentnumber==1):
        finalNeibor=ExtentOneCircle(A);
    elif(extentnumber==2):
        finalNeibor=ExtentOneCircle(ExtentOneCircle(A));
    elif(extentnumber==3):
        finalNeibor=ExtentOneCircle(ExtentOneCircle(ExtentOneCircle(A)));
    elif(extentnumber==4):
        finalNeibor=ExtentOneCircle(ExtentOneCircle(ExtentOneCircle(ExtentOneCircle(A))));
    return finalNeibor;

##对于多波段，外围扩充1，2，3，4圈
def MubandsWindowSizeEX(Mubands,extentnumber):
    numberBands=Mubands.shape[2];
    temp = WindowSizeFunction(Mubands[:,:,0],extentnumber);
    for ii in np.arange(1,numberBands):
        temp = np.dstack((temp, WindowSizeFunction(Mubands[:,:,ii],extentnumber) ));
    output=temp;
    return output;

#根据样本点栅格数据，找出非0值的像素对应的行列号，并找到对应位置的影像像素值（作为finalSampleX）
def MakeSample(ALLXfeature,allRowCols,Ylable,EXTnumber):
    hang = ALLXfeature.shape[0];  
    lie = ALLXfeature.shape[1];  NumberBands = ALLXfeature.shape[2];
    LENTH = len(allRowCols);  
    SamplePathes = [];    
    if EXTnumber >= 1:
        afterExtent = MubandsWindowSizeEX(ALLXfeature,EXTnumber);
        useYlabel = [];
        for ii in range(LENTH):
            hang = allRowCols[ii][0]+EXTnumber;    lie = allRowCols[ii][1]+EXTnumber;
            onePath = afterExtent[hang-EXTnumber:hang+EXTnumber+1,lie-EXTnumber:lie+EXTnumber+1,:];
            if (onePath.shape[0] == onePath.shape[1]):
                SamplePathes.append(  onePath.flatten() );
                useYlabel.append( Ylable[ii] );
        tempX = np.array(SamplePathes);
        tempY = np.array( useYlabel );
        
    else:   # EXTnumber<1,则不考虑空间邻域信息  
        AllXdata = ALLXfeature.reshape(hang*lie,NumberBands);
        SampleIndex = [];       
        for temp in allRowCols:
            SampleIndex.append(temp[0]*lie+temp[1]);
        tempX = AllXdata[SampleIndex,:];
        tempY = np.array(Ylable);
    return tempX,tempY;

#  样本数据划分 ，按一定比例分出训练集和测试集 
def SplitSample(Xdata,Ylabelsample,scale):
    X_train,X_test,y_train,y_test = train_test_split (Xdata,Ylabelsample,
                                    test_size = scale, random_state=2,
                                    stratify = Ylabelsample);
    return X_train,X_test,y_train,y_test;


#多类别分类结果绘图
def Draw2orMoreClassResult(PYLabel,titleName):
    #天汇数据，1为水体，2为草地，3为建筑，4为道路，5为裸地，6为背景
    ColorList = ['BLue', 'Yellow'];
    #TH6color = ['BLue','Green','Red','Yellow','Gray','Black' ];
    cor = ListedColormap(ColorList);
    plt.figure(figsize=(20,20));
    plt.imshow(PYLabel,cmap=cor); 
    plt.xticks(fontsize=20);   plt.yticks(fontsize=20);
    #plt.colorbar();
    plt.title(titleName,fontsize = 30);


#多类别分类最终类别判定
def finalLable2orMoreClass(predictY):
    finalY= 0*np.arange(0,predictY.shape[0]);
    for jj in range(predictY.shape[0]):
        temp = predictY[jj];
        index = np.where( temp == np.max(temp) )
        indexArray = list(index) 
        finalY[jj] = indexArray[0][0] + 1;
    return finalY;

