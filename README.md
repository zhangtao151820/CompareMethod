# CompareMethod
A lightweight [CompareMethod](https://www.mdpi.com/2072-4292/11/1/2) sklearn and keras implementation.

## Research area and samples
![](https://github.com/zhangtao151820/CompareMethod/blob/master/results/Figure%201.jpg)

## Landsat 8 - OLI Image
The image comes from GEE.(Google Earth Engine:https://code.earthengine.google.com/)

## Code description

The folder named "FeatureEngineering" includes all the code of SVM,RF,AdaBoost based on single pixels and image patches.

The folder named "FeatureLearning" includes all the code of CNN based on single pixels and image patches.

## Results display

Classification results based on feature engineering:

![](https://github.com/zhangtao151820/CompareMethod/blob/master/results/Figure%207.jpg)

Classification results based on feature learning:

![](https://github.com/zhangtao151820/CompareMethod/blob/master/results/Figure%209.jpg)


## Requirements
- Python 3.6 (didn't try 2.7)
- Tensorflow
- keras
- skleran
- Matplotlib