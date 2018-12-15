# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 12:31:00 2018

@author: zhang
"""
try:
    from osgeo import gdal  
    from osgeo import ogr  
except ImportError:
    import gdal  
    import ogr
from osgeo import osr
import math

import numpy as np
from skimage import io
import scipy.io as sio
import scipy.ndimage as SNDIG
from sklearn import preprocessing

#单个波段最近邻上采样，分辨率从30变为15米
def upsampleOneBand(inputsample,scaleX,scaleY):
    out = SNDIG.zoom(inputsample, (scaleX,scaleY), order=0);
    return out;

#多光谱数据（8个波段）最近邻上采样，分辨率从30变为15米
def MuBandsUp(inputsample,scaleX,scaleY):
    outputSample=[];  number=inputsample.shape[2];
    tempdstack = SNDIG.zoom(inputsample[:,:,0], (scaleX,scaleY), order=0);
    for ii in np.arange(1,number):
        tempdstack = np.dstack((tempdstack,SNDIG.zoom(inputsample[:,:,ii], (scaleX,scaleY), order=0) ));
    outputSample = tempdstack;
    return np.array(outputSample);

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

# 只有原始光谱 （ 多光谱 B1-B7 + 全色 ）
def ReadImageBands(panpath,multyBands):       
    Imagepan = io.imread(panpath);    # 全色
    bands = io.imread(multyBands);    # 多光谱
    Imagebands = MuBandsUp(bands,2,2);  #最近邻上采样为15米
    return np.dstack((Imagebands,Imagepan));

# 建筑物遥感指数
def BuiltupRSindex(panpath,builtUpIndex):
    Imagepan = io.imread(panpath);    # 全色
    BuiltIndex = sio.loadmat(builtUpIndex)['BI'];    # NDBI和IBI
    Imagebands = MuBandsUp(BuiltIndex,2,2);  #最近邻上采样为15米
    NDBIandIBI = Normalized( Imagebands );
    return np.dstack(( Imagepan,NDBIandIBI ));
    

# 纹理特征(PanTex、ASM、CORR、ENT、IDW)
def TextureFive(PanTexpath, ASMpath, CORRpath, ENTpath, IDWpath):
    def FilterPanTex(CONdata):
        CONdata[CONdata >= 500] = 500; return CONdata;
    def ReplaceNAN(data):
        data[np.isnan(data)] = 0;  data[np.isinf(data)] = 0;  return data;
    PanTex =ReplaceNAN( FilterPanTex( io.imread(PanTexpath) ));
    ASM = ReplaceNAN( io.imread(ASMpath));
    CORR = ReplaceNAN(io.imread(CORRpath));
    ENT = ReplaceNAN(io.imread(ENTpath));
    IDW = ReplaceNAN(io.imread(IDWpath));
    return np.dstack(( Normalized(PanTex), Normalized(ASM), Normalized(CORR),Normalized(ENT),Normalized(IDW) ));
    #return np.dstack(( PanTex, ASM, CORR,ENT,IDW ));

# 纹理特征(全色+PanTex)
def PanTexIndex(panpath,PanTexpath):
    def FilterPanTex(CONdata):
        CONdata[CONdata >= 500] = 500; return CONdata;
    def ReplaceNAN(data):
        data[np.isnan(data)] = 0;  data[np.isinf(data)] = 0;  return data;
    Imagepan = io.imread(panpath);
    PanTex = ReplaceNAN( FilterPanTex( io.imread(PanTexpath) ));
    return np.dstack((Imagepan, Normalized(PanTex) ));
                      
# 形态学建筑指数(EMBI)
def EMBImat(panpath,EMBIpath):
    Imagepan = io.imread(panpath);    # 全色
    EMBI = sio.loadmat(EMBIpath)['EMBIbuildIMG'];
    #STDEMBI = preprocessing.StandardScaler().fit_transform(EMBI);
    return np.dstack(( Imagepan, Normalized(EMBI) ));

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
        Row =int(math.floor( tempRowCol[1] ));  Col =int( math.floor( tempRowCol[0] ));
        RowsCols.append( [Row,Col] );
    return RowsCols,allLabel;
#----------------------------------------------------------------------------------#




