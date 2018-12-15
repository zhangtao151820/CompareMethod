# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 22:16:51 2018

@author: zhang
"""
# from sklearn.model_selection import StratifiedKFold       # 分层 k 折交叉验证
# from sklearn.model_selection import KFold                #  k 折交叉验证
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV  # 超参数网格搜索和随机搜索 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import PrepareSampleXData as PPSXD


# 固定参数的分类器
def DeterminedClassifier(X_train,y_train,number):
    def DefineClassifer():
        InitclfRFC = RandomForestClassifier(n_estimators=60);
        AdaBoostClf = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=60,learning_rate = 10e-3 );
        #SVMclf = SVC(C = 10,kernel = 'rbf');  # ,gamma = 2
        SVMclf = SVC(C = 10,kernel = 'rbf',gamma = 100);  # ,gamma = 2     3*3，5*5，7*7邻域，将惩罚系数C设为1
        mlpclf = MLPClassifier(hidden_layer_sizes=(8,6,6),activation='relu',solver='lbfgs',
                  alpha = 0.001,learning_rate='invscaling',random_state=4, max_iter=2000);
        return [InitclfRFC,AdaBoostClf,SVMclf,mlpclf];
    myModel = DefineClassifer()[number];
    myClf = myModel.fit(X_train,y_train);
    return myClf;
    
# 网格搜索超参数设置
def SetallParam():
    RFparam_grid = {'n_estimators': np.arange(10,30,5),'criterion':['gini','entropy'],
                    'max_depth':np.arange(10,20,5)};
    AdaBoosparam_grid = {'n_estimators': np.arange(10,30,2), 'learning_rate':[1.0,10e-1,10e-2,10e-3,10e-4],
                         'algorithm':['SAMME','SAMME.R'] };
    SVMparam_grid = {'C': [10e-1,1.0,10,10e2,10e3,10e4],
                     'kernel':['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'] };
    MLPparam_grid = {'learning_rate_init': [10e-2,10e-3,10e-4,10e-5],
                    'batch_size':np.arange(20,120,20)};
    return [RFparam_grid,AdaBoosparam_grid,SVMparam_grid,MLPparam_grid];

#  初始化分类器
def  InitializeClassifer():
    InitclfRFC = RandomForestClassifier();
    AdaBoostClf = AdaBoostClassifier(DecisionTreeClassifier() );
    SVMclf = SVC();
    mlpclf = MLPClassifier(hidden_layer_sizes=(8,6,6),activation='relu',solver='lbfgs',
             alpha=0.001,learning_rate='invscaling',random_state=4, max_iter=2000);
    return [InitclfRFC,AdaBoostClf,SVMclf,mlpclf];


# 使用训练好的模型进行分类，每次只对一行数据进行分类
import numpy as np
def UseModelToPredict(MYmodel,allFeature,EXTnumber):
    Rows = allFeature.shape[0];  Cols = allFeature.shape[1]; 
    resultLabel = np.ones((Rows,Cols), dtype=np.uint8)*0 + 4;
    # EXTnumber<1,则不考虑空间邻域信息
    if EXTnumber >= 1:
        afterExtent = PPSXD.MubandsWindowSizeEX(allFeature,EXTnumber);
        for index in np.arange(0,Rows):
            Hindex = index+EXTnumber;
            HangData = afterExtent[Hindex-EXTnumber:Hindex+EXTnumber+1,:,:];
            HangPathes = [];
            for kk in np.arange(0,Cols):
                Lind = kk + EXTnumber;
                onepath =  HangData[:,Lind-EXTnumber:Lind+EXTnumber+1,:];
                HangPathes.append( onepath.flatten() );
            HangResult = MYmodel.predict( np.array(HangPathes) );
            resultLabel[index,:] = HangResult.flatten();
    else:
        for index in np.arange(0,Rows):
            tempHang = allFeature[index,:,:];
            HangResult = MYmodel.predict(tempHang);
            resultLabel[index,:] = HangResult.flatten();
    return resultLabel;

# =====================================网格搜索===========================================
def GridSearchCVforBestParameter(X_train,y_train,number):
    #针对每个参数对进行了k(k=3)次交叉验证。scoring='accuracy'使用准确率为结果的度量指标。可以添加多个度量指标
    myclassfier = InitializeClassifer()[number];
    allparam = SetallParam()[number];
    grid = GridSearchCV(estimator = myclassfier, param_grid = allparam,
                        cv=3,n_jobs=-1, scoring='accuracy');
    grid.fit(X_train, y_train);
    bestModel = grid.best_estimator_;
    print(grid.best_params_);   print(grid.best_score_);
    return bestModel;

# =====================================随机搜索===========================================
def RandomizedSearchCVforBestParameter(X_train,y_train,number):
    myclassfier = InitializeClassifer()[number];
    allparam = SetallParam()[number];
    rand = RandomizedSearchCV(estimator = myclassfier, param_distributions = allparam,
                              cv= 3, n_jobs=-1,scoring='accuracy', n_iter = 10, random_state=5);
    rand.fit(X_train, y_train);
    bestModel = rand.best_estimator_;
    print(rand.best_params_);     print(rand.best_score_);
    return bestModel;