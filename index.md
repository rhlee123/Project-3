# Midterm Project 
## Introduction 
  The goal of this project is to evaluate the performance of different regressors when predicting the price of a property using the features within the boston housing data set, as well as polynomial features created from the original data. First, I explored the use of regularized regression on polynomial features created from the original data to predct housing prices in boston. In particular I usied six different regularization techniques: LASSO, Ridge, Elastic Net, SCAD (Smoothly-Clipped Absolute Deviation), Square Root Lasso, and Stepwise. Further I explored the use kernel regression, Random Forest, XGBoost, and Neural networks. I used K-fold cross validation to examine each model's mean absolute error. The model that yields the lowest mean absolute error when predicting the price would be the most accurate. 
## Start 
  Below, I imported relevant packages and pulled in the data set. I also set the parameter value k equal to 10 for my K-fold cross validation. K-fold cross validation shuffles the data set, splits the data sets into k groups (10 in my case), evaluates each group as its own individual test set and using the remaining groups as a training set to fit the model, and retains an evaluation score for each group. Ultimately, the accuracy of a model is summarized by the sample of the model evaluation scores collected for each group.
```python 
import numpy as np
import pandas as pd
from math import ceil
from scipy import linalg
from scipy.interpolate import interp1d
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.model_selection import train_test_split as tts
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy.optimize import minimize
from scipy.linalg import toeplitz
import operator
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from statsmodels.nonparametric.kernel_regression import KernelReg
kf = KFold(n_splits=k,shuffle=True,random_state=1234)
df_boston = pd.read_csv('/content/gdrive/MyDrive/Colab Notebooks/Boston Housing Prices(1) (1).csv')
```
Preprocessing: 
```python 
df_boston
features = ['crime','rooms','residential','industrial','nox','older','distance','highway','tax','ptratio','lstat']
X = np.array(df_boston[features])
y = np.array(df_boston['cmedv']).reshape(-1,1)
```

To take a look at the boston housing data set, here is a heatmap showing the correlations between features in the dataset in which we will be using to predict housing prices: 

![project1](https://user-images.githubusercontent.com/55299814/111015985-6133d380-8379-11eb-9c51-925a01550166.png)

