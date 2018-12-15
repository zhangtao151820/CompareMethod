# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 21:23:23 2018

@author: zhang
"""
try:
    from osgeo import gdal  
    from osgeo import ogr  
except ImportError:
    import gdal  
    import ogr
from osgeo import osr
#from ospybook as pb
#import scipy.io as sio
import scipy.ndimage as SNDIG
import keras,math
import numpy as np
from skimage import io
#from sklearn.feature_extraction import image
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Normalized 将输入特征数据进行拉伸至0-255
def Normalized(Onefeature):
    L= len(Onefeature.shape);
    if(L<=2):
        minx=np.min(Onefeature);   maxX=np.max(Onefeature);
        outtempX = 255*(Onefeature-minx)/(maxX-minx); 
        outX = outtempX.astype(np.uint8);
    else:
        outX=Onefeature;
        for index in range(Onefeature.shape[2]):
            tempX=Onefeature[:,:,index];
            minx = np.min(tempX);   maxX = np.max(tempX);
            outtempX = 255*(tempX-minx)/(maxX-minx);
            # INTout = map(int,outtempX);
            outX[:,:,index] = outtempX.astype(np.uint8);
    return outX;

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

#多光谱数据（8个波段）最近邻上采样，分辨率从30变为15米
def MuBandsUp(inputsample,scaleX,scaleY):
    outputSample=[];  number=inputsample.shape[2];
    tempdstack = SNDIG.zoom(inputsample[:,:,0], (scaleX,scaleY), order=0);
    for ii in np.arange(1,number):
        tempdstack = np.dstack((tempdstack,SNDIG.zoom(inputsample[:,:,ii], (scaleX,scaleY), order=0) ));
    outputSample = tempdstack;
    return np.array(outputSample);

# 全色 + 前三个主成分
def ReadPCA3orRGB(PCApath):
    bands = io.imread(PCApath);    # 多光谱
    return bands;


# 只有原始光谱 （全色 + 多光谱 B1-B7）
def ReadImageBands(panpath,multyBands):       
    Imagepan = io.imread(panpath);    # 全色
    bands = io.imread(multyBands);    # 多光谱
    Imagebands = MuBandsUp(bands,2,2);  #最近邻上采样为15米
    return np.dstack((Imagepan,Imagebands));

#---------------训练和测试样本 Label------------------------------------------------------#
#获取每个样本点shape的所有点的坐标,并转为全色影像对应的行列号
def GetXYValueToRowsCols(PanPath,shapePath):
    #获得给定数据的投影参考系和地理参考系
    def getSRSPair(dataset):
        prosrs = osr.SpatialReference()
        prosrs.ImportFromWkt(dataset.GetProjection())
        geosrs = prosrs.CloneGeogCS()
        return prosrs, geosrs;
    #将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
    def lonlat2geo(dataset, lon, lat):
        prosrs, geosrs = getSRSPair(dataset);
        ct = osr.CoordinateTransformation(geosrs, prosrs);
        coords = ct.TransformPoint(lon, lat);
        return coords[:2];
    #根据GDAL的六参数模型将给定的投影转为影像图上坐标（行列号）
    def geo2imagexy(dataset, x, y):
        trans = dataset.GetGeoTransform();
        a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]]);
        b = np.array([x - trans[0], y - trans[3]]);
        return np.linalg.solve(a, b);  # 使用numpy的linalg.solve进行二元一次方程的求解
    #获取每个样本点shape的所有点的坐标,并转为全色影像对应的行列号
    ogr.RegisterAll();  # 注册所有的驱动
    ds = ogr.Open(shapePath,0);      lyr = ds.GetLayer(0);     
    DSpan = gdal.Open(PanPath);        # print(DSpan.GetProjection());
    RowsCols = [];       allLabel = [];
    for feat in lyr:
        pt = feat.geometry();    OneLabel = feat.GetField('LC'); 
        allLabel.append( int(OneLabel) );
        CoordsXY = lonlat2geo(DSpan,pt.GetX(),pt.GetY()); 
        tempRowCol = geo2imagexy(DSpan, CoordsXY[0], CoordsXY[1]);
        Row = int(math.floor( tempRowCol[1] ));  Col = int( math.floor( tempRowCol[0] ));
        RowsCols.append( [Row,Col] );
    return RowsCols,np.array(allLabel);

#根据行列号制作所需的样本(考虑5*5邻域，对于keras的网络框架，输入的patch尺寸为5*5*8) EXTnumber=2
def MakeSample(allRowCols,allLabel, IsImagenet, allImageData,EXTnumber,scale):
    #Nubands = allImageData.shape[2];
    afterExtent = MubandsWindowSizeEX(allImageData,EXTnumber);  #原始多波段数据外围进行扩展
    #以邻域窗口进行滑动，获取所有的patch
    #patches = image.extract_patches_2d(afterExtent, (2*EXTnumber+1, 2*EXTnumber+1) ); 
    #训练样本的patch
    LENTH = len(allRowCols);        #WS = 2*EXTnumber+1;   
    #SamplePathes = np.zeros((LENTH,WS, WS,Nubands),dtype = np.uint16);
    SamplePathes = [];              finalUseLabel = [];
    for ii in range(LENTH):
        if IsImagenet==True:
            if (ii % 1)==0:
                hang = allRowCols[ii][0]+EXTnumber;           lie = allRowCols[ii][1]+EXTnumber;
                onePath = afterExtent[hang-EXTnumber:hang+EXTnumber+1,lie-EXTnumber:lie+EXTnumber+1,:];
                if (onePath.shape[0] == onePath.shape[1]):
                    SamplePathes.append( MuBandsUp(onePath,scale,scale) );
                    finalUseLabel.append( allLabel[ii] );
        else:
            hang = allRowCols[ii][0]+EXTnumber;    lie = allRowCols[ii][1] + EXTnumber;
            onePath = afterExtent[ hang-EXTnumber:hang+EXTnumber+1, lie-EXTnumber:lie+EXTnumber+1, :];
            if (onePath.shape[0] == onePath.shape[1]):
                SamplePathes.append( onePath );
                finalUseLabel.append( allLabel[ii] );        
    return np.array(SamplePathes),  np.array(finalUseLabel);


#根据行列号制作所需的样本(单像素，对于keras的网络框架，输入尺寸为1*8)
def MakeOnePixelSamples(allRowCols,allLabel,allImageData):
    Rows = allImageData.shape[0];   Cols = allImageData.shape[1];   BUb = allImageData.shape[2];
    LENTH = len(allRowCols);   SamplePathes = [];   finalUseLabel = [];
    for ii in range(LENTH):
        hang = allRowCols[ii][0];    lie = allRowCols[ii][1] ;
        if (  hang < Rows and lie < Cols ):
            onePixel = allImageData[ hang, lie, :];  
            SamplePathes.append( onePixel.reshape(BUb,1) );
            finalUseLabel.append( allLabel[ii] );
    return np.array(SamplePathes),  np.array(finalUseLabel);



#  样本数据划分 ，按一定比例分出训练集和测试集
def SplitSample(Xdata,Ylabelsample,scale,numberClass):
    X_train,X_test,y_train,y_test = train_test_split (Xdata,Ylabelsample,
                                    test_size = scale, random_state=2,
                                    stratify = Ylabelsample);
    y_trainfinal = keras.utils.to_categorical(y_train-1, num_classes = numberClass); 
    y_testfinal = keras.utils.to_categorical(y_test-1, num_classes = numberClass);                                                
    return X_train,X_test,y_trainfinal,y_testfinal;

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

#主成分分析，取前三个波段
from sklearn.decomposition import PCA
def PCATransform3(ImageData):
    hang = ImageData.shape[0];  
    lie = ImageData.shape[1];  NumberBands = ImageData.shape[2]; 
    AllXdata = ImageData.reshape(hang*lie,NumberBands);
    pcaUse = PCA(n_components = 3).fit(AllXdata);
    XafterTransform = pcaUse.transform(AllXdata);
    output = XafterTransform.reshape(hang,lie,3);
    return Normalized(output);

