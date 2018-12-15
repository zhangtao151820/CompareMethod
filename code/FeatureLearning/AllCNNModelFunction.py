# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 10:35:19 2018

@author: zhang
"""
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout,Flatten,Conv3D
from keras.layers import Conv2D, MaxPooling2D,regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv1D,Concatenate,Input,Reshape,Lambda,merge

#---------------一维卷积核CONV1D ----------------------------------------------#
def Conv1DdefCNN(NumberBands,numberClass):
    model = Sequential()
    #filter的深度为32,对每个像素在深度方向卷积,默认步长为 1
    #model.add( Reshape((NumberBands,1), input_shape=(1,NumberBands)) )
    model.add(Conv1D(256, kernel_size= 3,  strides=1, dilation_rate=2,
                     activation='relu',input_shape=(NumberBands,1) ));  
    #filter的深度为64,对每个像素在深度方向卷积
    model.add(Conv1D(512, kernel_size = 4,strides=1, activation='relu')); 
    model.add(Flatten());  #转成一维向量
    model.add(Dense(256, activation='relu'));
    model.add(Dropout(0.25));
    model.add(Dense(512, activation='relu'));  
    model.add(BatchNormalization() );
    model.add(Dense(128, activation='relu')); 
    model.add(Dense(numberClass, activation='softmax')); #softmax
    return model;
#--------------------------------------------------------------------------------#


#---------------二维卷积核CONV2D  （两层Conv2D+三层Dense)------------------------------#
def Conv2DCNN(Winsize,NumberBands,numberClass):
    model = Sequential();
    #filter的深度为32,,卷积后输出为32层，5*5
    model.add(Conv2D(256, (3, 3),padding='same',activation='relu', 
                 input_shape=(Winsize, Winsize, NumberBands)));  
    model.add(MaxPooling2D( pool_size=(3, 3), strides=(1,1) ));            
    model.add(Conv2D(512, (3, 3),padding='same', activation='relu' ));
    model.add(MaxPooling2D( pool_size=(3, 3), strides=(1,1) )); 
    model.add(BatchNormalization() ); 
    #model.add(MaxPooling2D(pool_size=(2, 2))); # 下采样-最大池化，输出为128层，1*1
    model.add(Flatten());  #转成一维向量
    model.add(Dense(256, activation='relu'));
    model.add(Dropout(0.25));
    model.add(Dense(512, activation='relu'));  #全连接层，64个神经元
    model.add(BatchNormalization() );
    model.add(Dense(128, activation='relu'));  #全连接层，32个神经元
    model.add(Dense(numberClass, activation='softmax')); #softmax
    return model;
# ---------------------------------------------------------------------------------------#
    

# --------------------先在一维方向卷积(波段运算)，然后在二维方向进行卷积-------------------#    
def FirstBandsThenNeiborCNN(Winsize,NumberBands,numberClass):
    def myslice(x,Row,Col):
        return x[:,Row,Col,:];
    inputOne = Input(shape=(Winsize,Winsize,NumberBands)); 
    allOut=[];       
    for ii in range(Winsize):
        ALLinner = [];
        for jj in range(Winsize):
            x1 = Lambda(myslice, output_shape=(1,1,NumberBands), arguments={'Row':ii,'Col':jj})(inputOne)
            x2 = Reshape((1,NumberBands))(x1) 
            first = Conv1D(256, kernel_size=1, strides=2,activation='relu')(x2);
            drop = Dropout(0.25)(first);
            Batch = BatchNormalization()(drop);
            second = Conv1D(512, 1,strides=3, activation='relu')(Batch);
            x3 = Reshape((1,1,512))(second);  # <tf.Tensor 'reshape_42/Reshape:0' shape=(?, 1, 1, 64) dtype=float32>
            ALLinner.append(x3);
        # 沿着第二个维度参数（按列）进行拼接，<tf.Tensor 'merge_5/concat:0' shape=(?, 1, 5, 64) dtype=float32>
        conInner = merge( ALLinner, mode='concat',concat_axis=2);
        allOut.append(conInner);
    outConcatenate = merge( allOut, mode='concat',concat_axis=1);
    
    Basedmodel = Model(inputs = inputOne, outputs=outConcatenate);  #  Basedmodel实现在波段方向的卷积
    x = Basedmodel.output
    x = Conv2D(512, (3, 3), activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(1024, (3, 3), activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization()(x);
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x);
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x);
    x = Dense(256, activation='relu')(x)
    # and a logistic layer -- let's say we have 2 classes
    predictions = Dense(numberClass, activation='softmax')(x)
    # this is the model we will train  #model以Basedmodel的输出作为输入，实现平面方向的卷积
    model = Model(inputs=Basedmodel.input, outputs=predictions);
    return model;
# ---------------------------------------------------------------------------------------#
    

# -------------------三维方向卷积Conv3D---------------------------------------------------#
def Conv3DCNN(Winsize,NumberBands,numberClass):
    model = Sequential();
    model.add(Conv3D(64, (3, 3, 2), strides=(1, 1, 2),activation='relu', 
                     input_shape=(Winsize, Winsize, NumberBands,1)));   
    model.add(Conv3D(128, (3, 3, 1),strides=(1, 1, 1), activation='relu')); 
    model.add( BatchNormalization() );
    model.add(Flatten());  #转成一维向量
    model.add(Dense(256, activation='relu'));
    model.add(Dropout(0.25));
    model.add(Dense(512, activation='relu'));  #全连接层，个神经元
    model.add(BatchNormalization() );
    model.add(Dense(128, activation='relu'));  #全连接层，个神经元
    model.add(Dense(numberClass, activation='softmax')); #softmax
    return model;
# ---------------------------------------------------------------------------------------#
    

#-----------deconvolution 反卷积网路，实现图像语义分割------------------------------------#
from keras.layers import UpSampling2D,Conv2DTranspose
def  DeconvolutionCNN():
    model = Sequential();
    #----------------------卷积部分-----------------------------------------------
    model.add(Conv2D(64, (3, 3),strides=(2, 2),activation='relu', input_shape=(200, 200, 3)));
    model.add(Conv2D(128, (3, 3),strides=(2, 2), activation='relu'));       
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)));  # pooling向下取样
    model.add(Conv2D(64, (3, 3), strides=(2, 2),activation='relu'));
    model.add(Conv2D(32, (3, 3), strides=(2, 2),activation='relu'));
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)));
    model.add(MaxPooling2D(pool_size=(5, 5)));
    #---------------------------------------------------------------------------------#
    #-------------------------反卷积部分----------------------------------------------#
    model.add(UpSampling2D(size=(5, 5)));
    model.add(UpSampling2D(size=(2, 2)));
    model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2)));
    model.add(Conv2DTranspose(128, (3, 3), strides=(2, 2)));
    model.add(UpSampling2D(size=(2, 2)));
    model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2)));
    model.add(Conv2DTranspose(1, (3, 3), strides=(2, 2)));
    #-------------------------------------------------------------------------------#
    return model;
# ---------------------------------------------------------------------------------------#


from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D

# -------------------------对VGG16模型进行微调---------------------------------------#
def VGG16Finetuning(WS):  # default the input shape  is (224, 224, 3)
    # It should have exactly 3 input channels,and width and height should be no smaller than 48.
    base_model = VGG16(weights='imagenet', include_top=False,input_shape=(WS,WS,3));
    for layer in base_model.layers:
        layer.trainable = False
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(64, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x);
    x = Dense(64, activation='relu')(x)
    # and a logistic layer -- let's say we have 2 classes
    predictions = Dense(2, activation='softmax')(x)
    # this is the model we will train
    mymodel = Model(inputs= base_model.input, outputs=predictions)
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional VGG16 layers
    return mymodel;
# ---------------------------------------------------------------------------------------#

# 对InceptionV3模型进行微调
def InceptionV3Finetune(WS):   #default input image size for this model is 299x299,3个波段
    #It should have exactly 3 inputs channels, and width and height should be no smaller than 139.
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False,input_shape=(WS,WS,3))
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(64, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x);
    x = Dense(64, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(2, activation='softmax')(x)
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model;
# ---------------------------------------------------------------------------------------#


# --------对ResNet50模型进行微调--------------------------------------------------#
def  ResNet50Finetune(WS):  #默认 input shape is (224, 224, 3)
    #It should have exactly 3 inputs channels, and width and height should be no smaller than 197.
    base_model = ResNet50(weights='imagenet', include_top=False,input_shape=(WS,WS,3));
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(64, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x);
    x = Dense(64, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(2, activation='softmax')(x)
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model;
# ---------------------------------------------------------------------------------------#




