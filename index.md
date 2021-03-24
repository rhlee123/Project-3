# Midterm Project 
## Introduction 
  The goal of this project is to evaluate the performance of different regressors when predicting the price of a property using the features within the boston housing data set, as well as polynomial features created from the original data. First, I explored the use of regularized regression on polynomial features created from the original data to predct housing prices in boston. In particular I usied six different regularization techniques: LASSO, Ridge, Elastic Net, SCAD (Smoothly-Clipped Absolute Deviation), Square Root Lasso. Further I explored the use of stepwise regression kernel regression, Random Forest, XGBoost, and Neural networks, and ultimately compared all models to a baseline linear regression model. I used K-fold cross validation to examine each model's mean absolute error. The model that yields the lowest mean absolute error when predicting the price would be the most accurate. 
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
Preprocessing as well as setting a standard scaler and the degree at which polynomial features will be created from using the original data (we will be using a degree of 3 for all models that will use polynomial features): 
```python 
df_boston
features = ['crime','rooms','residential','industrial','nox','older','distance','highway','tax','ptratio','lstat']
X = np.array(df_boston[features])
y = np.array(df_boston['cmedv']).reshape(-1,1)
scale = StandardScaler()
poly = PolynomialFeatures(degree=3)
```

To take a look at the boston housing data set, here is a heatmap showing the correlations between features in the dataset in which we will be using to predict housing prices: 

![project1](https://user-images.githubusercontent.com/55299814/111015985-6133d380-8379-11eb-9c51-925a01550166.png)
## Linear Regression 
Linear regressions model the linear relationship between a dependent variable and independent variable(s). This method calculates the best-fitting line for the observed data by minimizing the sum of the squares of the vertical deviations from each data point to the line (also known as the Sum of Squared Residuals or the Sum of Squared Errors). Below I fit the linear regression and found the mean absolute error of the model's predictions using K-fold cross validation.
```python 
lm = LinearRegression()
mae_lm = []

for idxtrain, idxtest in kf.split(X):
    X_train = X[idxtrain,:]
    y_train = y[idxtrain]
    X_test  = X[idxtest,:]
    y_test  = y[idxtest]
    lm.fit(X_train,y_train)
    yhat_test = lm.predict(X_test)
    mae_lm.append(MAE(y_test,yhat_test))
print("Validated MAE Linear Regression = ${:,.2f}".format(1000*np.mean(mae_lm)))
``` 
## Polynomial Regression 
  Using polynomial features created from our original dataset help when our features in our data do not fit a linear relationship well and would better fit a polynomial relationship. In this project, I will be using polynomical features with degree 3 for my multivariate regularized regressions and stepwise regression.
## Regularization 
  Regularization methods are used to determine the weights for features within a model, and depending on the regularization technique, features can be excluded altogether from the model by having a weight of 0. Further, regularization is the process of of regularizing the parameters that constrains or coefficients estimates towards zeros, in otherwords discouraging learning a too complex or too flexible model, and ultimately helping to reduce the risk of overfitting. Regularization helps to choose the preferred model complexity, so that the model is better at predicting. Regularization is essentially adding a penalty term to the objective function, and controlling the model complexity using that penalty term. Regularization ultmately attempts to reduce the variance of the estimator by simplifying it, something that will increase the bias, in such a way that will decrease the expected error of the model's predictions. Additionally, Regularization is useful in tackling issues of multicolinearity among features becauase it incorporates the need to learn additional information for our model other than the observations in the data set.
  
  Regularization techniques are especially useful in situations where there are large number of features or a low number of observations to number of features, where there is multicolinearity between the features, trying to seek a sparse solution, or accounting for for variable groupings in high dimensions. Regularization ultimately seeks to tackle the shortcomings of Ordinary Least Square models. In our case, it can be clearly seen that there is strong correlation between the features, which indicates that regularization might be helpful effectively estimating coefficients despite the multicolinearity present. Further, it indicates that regularization might reduce the MAE of a model that looks to predict price using these features. 
  
### Ridge 
L2 regularization, or commonly known as ridge regularization, is a type of regularization that controls for the sizes of each coefficient or estimators. Not only does ridge regularization encorporate principles of OLS by reducing the sum of squared residuals, it also penalizes models with the regularization term of L2 Norm or
![\alpha \sum_{i=1}^p \beta_i^2](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Calpha+%5Csum_%7Bi%3D1%7D%5Ep+%5Cbeta_i%5E2).

Ridge regularization is especially useful when there is multicolinearity within data, and further, ridge regularization seeks to ultimately minimze the cost function:

![\sum_{i=1}^N (y_i - \hat{y}_i)^2 + \alpha \sum_{i=1}^p |\beta_i^2| ](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%21%5B%5Csum_%7Bi%3D1%7D%5EN+%28y_i+-+%5Chat%7By%7D_i%29%5E2+%2B+%5Calpha+%5Csum_%7Bi%3D1%7D%5Ep+%7C%5Cbeta_i%5E2%7C+)
(α) in this instance is the hyperparameter that determines the strength of regularization, or the strength of the penalty on the model. 
### LASSO
L1 regularization, or commonly known as Least Absolute Shrinkage and Selection Operator (LASSO) regularization, determines the weight of features by penalizing the model with the regularization term of L1 Norm or ![\alpha \sum_{i=1}^p |\beta_i| ](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Calpha+%5Csum_%7Bi%3D1%7D%5Ep+%7C%5Cbeta_i%7C+). 

Further, LASSO regularization seeks to ultimately minimze the cost function: 

![Capture](https://user-images.githubusercontent.com/55299814/111152630-4f178800-8567-11eb-9f12-49e834aa0f2a.PNG)

LASSO differs from ridge in that the way that LASSO penalizes high coefficients, as instead of squaring coefficients, LASSO takes the absolute value of the coefficients. Ultimately the weights of features can go to 0 using L1 norm, as opposed to L2 norm that ridge regularization in which weights can not go to 0. Ridge regularization will shrink the coefficients for least important features, very close to zero, however, will never make them exactly zero, resulting in the final model including all predictors. However, in the case of the LASSO, the L1 norm penalty has the eﬀect of forcing some of the coeﬃcient estimates to be exactly equal to zero when the tuning parameter (α) is suﬃciently large. Therefore, the lasso method, not only performs variable selection but is generally said to yield sparse models.
### Elastic Net
Elastic net regularization combines aspects of both ridge and LASSO regularization by including both L1 norm and L2 norm penalties. Elastic net determines the weights of features by minimizing the cost funciton where λ between 0 and 1: 

![\hat{\beta} = argmin_\beta \left\Vert  y-X\beta \right\Vert ^2 + \lambda_2\left\Vert  \beta \right\Vert ^2 + \lambda_1\left\Vert  \beta\right\Vert_1
![jkiyutyfcgvhbkjnkihyguh](https://user-images.githubusercontent.com/55299814/111017541-3d28c000-8382-11eb-9681-03df13e00f9f.png)
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Chat%7B%5Cbeta%7D+%3D+argmin_%5Cbeta+%5Cleft%5CVert++y-X%5Cbeta+%5Cright%5CVert+%5E2+%2B+%5Clambda_2%5Cleft%5CVert++%5Cbeta+%5Cright%5CVert+%5E2+%2B+%5Clambda_1%5Cleft%5CVert++%5Cbeta%5Cright%5CVert_1%0A) 

Elastic net regularization is a good middle ground between the other techniques, ridge and LASSO, because the technique allow for the model to learns weights that fit the multicolinearity and sparsity pattern within the data.
#### Implementation
Setting up code to find K-fold cross validated MAE for LASSO, Ridge, Elastic Net regularized regressions: 
```python 
def DoKFold_SK(X,y,model,k):
  PE = []
  pipe = Pipeline([('scale',scale),('polynomial features',poly),('model',model)])
  for idxtrain, idxtest in kf.split(X):
    X_train = X[idxtrain,:]
    y_train = y[idxtrain]
    X_test  = X[idxtest,:]
    y_test  = y[idxtest]
    pipe.fit(X_train,y_train)
    yhat_test = pipe.predict(X_test)
    PE.append(MAE(y_test,yhat_test))
  return 1000*np.mean(PE)
```
### Square Root LASSO 
Square Root LASSO slightly adjusts the LASSO method, in which it takes the square root of the LASSO cost function. It is important to note that L1 norm is still used for its penalty. Ultimately, lasso weights features by minimizing the cost function: 

![\sqrt{\frac{1}{n}\sum\lim_{i=1}^{n}(y_i-\hat{y}_i)^2} +\alpha\sum\lim_{i=1}^{p}|\beta_i|
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Csqrt%7B%5Cfrac%7B1%7D%7Bn%7D%5Csum%5Clim_%7Bi%3D1%7D%5E%7Bn%7D%28y_i-%5Chat%7By%7D_i%29%5E2%7D+%2B%5Calpha%5Csum%5Clim_%7Bi%3D1%7D%5E%7Bp%7D%7C%5Cbeta_i%7C%0A)
#### Implementation 
Setting up code to find K-fold cross validated MAE for Square Root LASSO regularized regression as well as initialize square root LASSO:
```python
def sqrtlasso_model(X,y,alpha):
  n = X.shape[0]
  p = X.shape[1]
  
  def sqrtlasso(beta):
    beta = beta.flatten()
    beta = beta.reshape(-1,1)
    n = len(y)
    return np.sqrt(1/n*np.sum((y-X.dot(beta))**2)) + alpha*np.sum(np.abs(beta))
  
  def dsqrtlasso(beta):
    beta = beta.flatten()
    beta = beta.reshape(-1,1)
    n = len(y)
    return np.array((-1/np.sqrt(n))*np.transpose(X).dot(y-X.dot(beta))/np.sqrt(np.sum((y-X.dot(beta))**2))+alpha*np.sign(beta)).flatten()
  b0 = np.ones((p,1))
  output = minimize(sqrtlasso, b0, method='L-BFGS-B', jac=dsqrtlasso,options={'gtol': 1e-8, 'maxiter': 1e8,'maxls': 25,'disp': True})
  return output.x
```
```python 
def DoKFoldSqrt(X,y,a,k,d):
  PE = []
  scale = StandardScaler()
  poly = PolynomialFeatures(degree=d)
  for idxtrain, idxtest in kf.split(X):
    X_train = X[idxtrain,:]
    X_train_scaled = scale.fit_transform(X_train)
    X_train_poly = poly.fit_transform(X_train_scaled)
    y_train = y[idxtrain]
    X_test  = X[idxtest,:]
    X_test_scaled = scale.transform(X_test)
    X_test_poly = poly.fit_transform(X_test_scaled)
    y_test  = y[idxtest]
    beta_sqrt = sqrtlasso_model(X_train_poly,y_train,a)
    n = X_test_poly.shape[0]
    p = X_test_poly.shape[1]
    yhat_sqrt = X_test_poly.dot(beta_sqrt)
    PE.append(MAE(y_test,yhat_sqrt))
  return 1000*np.mean(PE)
```
### SCAD
The Smoothly Clipped Absolute Deviation regularization attempts to address issues of multicolinearity and encourage sparse solutions to ordinary least squares, while at the same time allowing for large (β) Values. 

The SCAD penalty is genearlly defined by it first derivative:

![p'_\lambda(\beta) = \lambda \left\{ I(\beta \leq \lambda) + \frac{(a\lambda - \beta)_+}{(a - 1) \lambda} I(\beta > \lambda) \right\}
](https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+p%27_%5Clambda%28%5Cbeta%29+%3D+%5Clambda+%5Cleft%5C%7B+I%28%5Cbeta+%5Cleq+%5Clambda%29+%2B+%5Cfrac%7B%28a%5Clambda+-+%5Cbeta%29_%2B%7D%7B%28a+-+1%29+%5Clambda%7D+I%28%5Cbeta+%3E+%5Clambda%29+%5Cright%5C%7D%0A)

with the penalty function represented by the piecewise function: 

![\begin{cases} \lambda & \text{if } |\beta| \leq \lambda \\ \frac{(a\lambda - \beta)}{(a - 1) } & \text{if } \lambda < |\beta| \leq a \lambda \\ 0 & \text{if } |\beta| > a \lambda \\ \end{cases}
](https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cbegin%7Bcases%7D+%5Clambda+%26+%5Ctext%7Bif+%7D+%7C%5Cbeta%7C+%5Cleq+%5Clambda+%5C%5C+%5Cfrac%7B%28a%5Clambda+-+%5Cbeta%29%7D%7B%28a+-+1%29+%7D+%26+%5Ctext%7Bif+%7D+%5Clambda+%3C+%7C%5Cbeta%7C+%5Cleq+a+%5Clambda+%5C%5C+0+%26+%5Ctext%7Bif+%7D+%7C%5Cbeta%7C+%3E+a+%5Clambda+%5C%5C+%5Cend%7Bcases%7D%0A)

The cost function ultimately looks like: 

![jkiyutyfcgvhbkjnkihyguh](https://user-images.githubusercontent.com/55299814/111017670-d7890380-8382-11eb-84e0-7e6908fb891a.png)

#### Implementation
Setting up code to find k-fold cross validated MAE for SCAD regularized regression as well as initialize SCAD:
```python 
def scad_penalty(beta_hat, lambda_val, a_val):
    is_linear = (np.abs(beta_hat) <= lambda_val)
    is_quadratic = np.logical_and(lambda_val < np.abs(beta_hat), np.abs(beta_hat) <= a_val * lambda_val)
    is_constant = (a_val * lambda_val) < np.abs(beta_hat)
    
    linear_part = lambda_val * np.abs(beta_hat) * is_linear
    quadratic_part = (2 * a_val * lambda_val * np.abs(beta_hat) - beta_hat**2 - lambda_val**2) / (2 * (a_val - 1)) * is_quadratic
    constant_part = (lambda_val**2 * (a_val + 1)) / 2 * is_constant
    return linear_part + quadratic_part + constant_part
    
def scad_derivative(beta_hat, lambda_val, a_val):
    return lambda_val * ((beta_hat <= lambda_val) + (a_val * lambda_val - beta_hat)*((a_val * lambda_val - beta_hat) > 0) / ((a_val - 1) * lambda_val) * (beta_hat > lambda_val))
    
def scad_model(X,y,lam,a):
  n = X.shape[0]
  p = X.shape[1]
  # we add aan extra columns of 1 for the intercept
  #X = np.c_[np.ones((n,1)),X]
  def scad(beta):
    beta = beta.flatten()
    beta = beta.reshape(-1,1)
    n = len(y)
    return 1/n*np.sum((y-X.dot(beta))**2) + np.sum(scad_penalty(beta,lam,a))
  
  def dscad(beta):
    beta = beta.flatten()
    beta = beta.reshape(-1,1)
    n = len(y)
    return np.array(-2/n*np.transpose(X).dot(y-X.dot(beta))+scad_derivative(beta,lam,a)).flatten()
  b0 = np.ones((p,1))
  output = minimize(scad, b0, method='L-BFGS-B', jac=dscad,options={'gtol': 1e-8, 'maxiter': 1e7,'maxls': 25,'disp': True})
  return output.x   
```
```python 
def DoKFoldScad(X,y,lam,a,k):
  PE = []
  for idxtrain, idxtest in kf.split(X):
    X_train = X[idxtrain,:]
    X_train_scaled = scale.fit_transform(X_train)
    X_train_poly = poly.fit_transform(X_train_scaled)
    y_train = y[idxtrain]
    X_test  = X[idxtest,:]
    X_test_scaled = scale.transform(X_test)
    X_test_poly = poly.fit_transform(X_test_scaled)
    y_test  = y[idxtest]
    beta_scad = scad_model(X_train_poly,y_train,lam,a)
    n = X_test_poly.shape[0]
    p = X_test_poly.shape[1]
    yhat_scad = X_test_poly.dot(beta_scad)
    PE.append(MAE(y_test,yhat_scad))
  return 1000*np.mean(PE)
```
### Stepwise Regression 
In the Stepwise regression, we focus on the the significance and influence of independent variable(features) on the dependent variable by using an approach that encorporates a combination of forward and backward variable selection techniques. Stepwise uses a threshold for the significance of a feature to determine whether or not that feature will be used as a variable. In practice, a threshold value for a p-value is incorporated such that features with p-values lower than the threshold will be encorporated into the model, while p-values higher than the threshold will be kicked out of the model. 
Code for intializing stepwise feature selection in which the output will be the indices of the columns for the variable that were 'selected' by the stepwise function (meet the p-value threshold) and ultimately incorporated into our linear model. After the outputs are found by inputing the polynomical features from the original data into the below function, we fit a multivariate linear model and perform k-fold cross validation using the features that were kept from our original polynomial features determined by the stepwise function.
#### Implementation
Code for implementing stepwise
```python 
def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.15, 
                       threshold_out = 0.05, 
                       verbose=True):    
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included
```
Using same process as cross-validating our baseline linear model, we can find the k-fold cross validated MAE for our stepwise regression:
```python 
lm = LinearRegression()
mae_lm = []
for idxtrain, idxtest in kf.split(X):
    X_train = X[idxtrain,:]
    y_train = y[idxtrain]
    X_test  = X[idxtest,:]
    y_test  = y[idxtest]
    lm.fit(X_train,y_train)
    yhat_test = lm.predict(X_test)
    mae_lm.append(MAE(y_test,yhat_test))
print("Validated MAE Linear Regression = ${:,.2f}".format(1000*np.mean(mae_lm)))
```
## Kernel Weighted Regression (Loess)
Kernels use their respective functions to determine the weights of our data points for our locally weighted regression. Kernel weighted regressions work well for data that does not show linear qualities. 
#### Implementation
Below I show how to implement the gaussian kernel:
```python 
def Gaussian(x):
  return np.where(np.abs(x)>1,0,np.exp(-1/2*x**2))
```
We will be using statsmodels kernelreg function for the gaussian kernel on our original dataset. Here is the code for finding k-fold cross validated MAE for guasian kernel regression on the original features:
```python 
mae_kern = []

for idxtrain, idxtest in kf.split(X):
  X_train = X[idxtrain,:]
  y_train = y[idxtrain]
  X_test  = X[idxtest,:]
  y_test  = y[idxtest]
  model_KernReg = KernelReg(endog=y_train,exog=X_train,var_type='ccccccccccc',ckertype='gaussian')
  yhat_sm_test, y_std = model_KernReg.fit(X_test)
  mae_kern.append(mean_absolute_error(y_test, yhat_sm_test))
print("Validated MAE Gaussian Kernel Regression = ${:,.2f}".format(1000*np.mean(mae_kern)))
```
## Neural Networks
Neural Networks are models that look to recognize underlying relationships in a data set through a process that is similar to the way that the human brain works. Neural networks use activation functions to transform inputs. In a neural network, a neuron is a mathematical function that collects and classifies information according to a specific structure or architecture. The neural network ultimately goes through a learning process in which it fine tunes the connection strengths and relationships between neurons in the network to optimize the neural networks performance in solving a particular problem, which in our case is predicting the price. 
#### Implementation
Below I created a function that uses K-fold validation to find the absolute mean error for the predictions of the neural network model as well as initialized the neural network model:
```python 
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import r2_score
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
model = Sequential()
model.add(Dense(128, activation="relu", input_dim=1))
model.add(Dense(32, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="linear"))
model.compile(loss='mean_absolute_error', optimizer=Adam(lr=1e-3, decay=1e-3 / 200))
```
``` python 
mae_nn = []

for idxtrain, idxtest in kf.split(dat):
  X_train = dat[idxtrain,:]
  y_train = dat[idxtrain,]
  X_test  = dat[idxtest,:]
  y_test = dat[idxtest,1]
  es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
  model.fit(X_train,y_train,validation_split=0.3, epochs=1000, batch_size=100, verbose=0, callbacks=[es])
  yhat_nn = model.predict(X_test.reshape(-1,1))
  mae_nn.append(mean_absolute_error(y_test, yhat_nn))
print("Validated MAE Neural Network Regression = ${:,.2f}".format(1000*np.mean(mae_nn)))
```
## XGBoost 
Boosting refers to a family of algorithms that look to turn weak learners into strong learners. In boosting, the individual models are built sequentially by putting more weight on instances where there are wrong predictions and high magnitudes of errors. The model will focus during learning on instances which are hard to predict correctly, so that the model in a sense learns from past mistakes. Extreme gradient boost is a decision-tree based algorithm that uses advanced gradient boosting and regularization to prevent overfitting. 
#### Implementation
I used K-fold validation to find the mean absolute error for the XGBoost model's predictions by running the below code:
```python 
model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=3)
mae_xgb = []

for idxtrain, idxtest in kf.split(X):
  X_train = X[idxtrain,:]
  y_train = y[idxtrain]
  X_test  = X[idxtest,:]
  y_test  = y[idxtest]
  model_xgb.fit(X_train,y_train)
  yhat_xgb = model_xgb.predict(X_test)
  mae_xgb.append(mean_absolute_error(y_test, yhat_xgb))
print("Validated MAE XGBoost Regression = ${:,.2f}".format(1000*np.mean(mae_xgb)))
```
## Random Forest 
Random Forests is a type of regression algorithm that expanands on the more basic Decision Trees algorithm. Random Forest algorithms create a 'forest' of Decision Trees that will be randomly sampled with replacement. Further, random forests will then give weights to each Decision Tree in order to control for some of the overfitting issues that are prevelant in normal Decision Tree algorithm. 
#### Implementation 
I used K-fold validation to find the mean absolute error for the Random Forest model's predictions by running the below code:
```python 
rf = RandomForestRegressor(n_estimators=1000,max_depth=3)
mae_rf = []

for idxtrain, idxtest in kf.split(X):
  X_train = X[idxtrain,:]
  y_train = y[idxtrain]
  X_test  = X[idxtest,:]
  y_test  = y[idxtest]
  rf.fit(X_train,y_train.ravel())
  yhat_rf = rf.predict(X_test)
  mae_rf.append(mean_absolute_error(y_test, yhat_rf))
print("Validated MAE Random Forest Regression = ${:,.2f}".format(1000*np.mean(mae_rf)))
```
## GridSearchCV 
The Grid Search algorithm is method that is useful in adjusting the parameters in supervised learning models and aim improve the performance of a model by selecting a competitive hyperparameter value. The Grid Search algorithm works by examining all possible combinations of the parameters of interest and finds the best ones. Although this algorithm aims to find the best hyperparameter that yields the lowest MAE, it does not necessarily always yield better results as some hyperparameters might be skipped over in our situation. I applied gridsearch on my models that have hyperparameters and found that gridsearch yielded hyperparameter values with worse MAE's than other methods of finding optimal alpha. Because gridsearch did not do very well, I ended up using another method of hypertuning my parameter.
#### Implementation
Example implementation of gridsearch: 
```python 
reg = LASSO()
params = [{'alpha':np.linspace(0.001,10,num=1000)}]
gs = GridSearchCV(estimator=reg,cv=10,scoring='neg_mean_absolute_error',param_grid=params)
gs1 = gs.fit(scale.fit_transform(X_poly),y)
print(gs1.best_params_)
print('The mean absolute error: ', np.abs(gs_results.best_score_)*1000)
```
## Hypertuning Parameters
Optimal hyperparameters found by plotting all instances of hyperparameter and respective MAE of predictions, and finding the global minimum. 
Here are optimal parameters found for our models:

| Model                          | Optimal α | Optimal Lambda|
|--------------------------------|-----------|---------------|                               
| MAE Ridge Regression Model     | 21        |               |             
| MAE LASSO Model                | .055      |               |             
| MAE Elastic Net Model          | .055      | .25           | 
| MAE SCAD Model                 | .125      | 5            |            
| MAE Square Root LASSO          | .009      | 1.25          | 

| Model                          | Optimal parameters|
|--------------------------------|---------------------------------------------------------|
| Neural Network Regression      | validation_split=0.3, epochs=1000, batch_size=100       | 
| XGBoost Regression             | n_estimators=100,lambda=20, alpha=1,gamma=10,max_depth=3|    
| Random Forest Regression       | n_estimators=100, max_depth=3                           |           
         
# Results 
| Model                          | Validated MAE | 
|--------------------------------|---------------|                               
| MAE Linear Regression          | $3,629.76     |                        
| MAE Ridge Model                | $2,187.97     | 
| MAE LASSO Model                | $2,211.85     |   
| MAE Elastic Net Model          | $2,169.88     | 
| MAE SCAD Model                 | $2,605.94     |            
| MAE Square Root LASSO          | $2,138.52     |
| MAE Stepwise Regression        | $3,512.45     | 
| MAE Gaussian Kernel Regression | $2,873.19     | 
| MAE Neural Network             | $2,475.77     | 
| MAE XGBoost                    | $2,313.58     | 
| MAE Random Forest              | $2,855.43     | 

# Conclusion 
From the results above, it can clearly be seen that all models used in the scenario produced better results (lower k-fold cross validated mean absolute errors) than just the baseline linear model. Further, in particular, it appears that our regularized models performed the best in this instance, and resulted in the lowest cross validated mean absolute errors as regularization helps in situations where there might be multicolinearity such as the boston housing data set used. In particular the square root LASSO regularized regression performed the best with the lowest cross validated mean absolute error of $2,138.52. Square root Lasso especially performed well in this situation because Square Root LASSO accounts for multicolinearity and penalizes in a way that produces sparse solutions. Further, Elastic net regularized regression also performed well with the second lowest validated MAE of $2,169.88. The performance of the elastic net model hints at the effectiveness of using a penalization that incorpoates both the L1 and L2 norm can be more effective than a model that only penalizes using either just l1 norm or just L2 norm. This can be seen as in this instance, elastic net regularized regression performed better than just a simple ridge or LASSO model (which elastic net is a combination of the two). Further it appears regularization techniques performed well not only because of the advantages of regularization but also because of the utilzation of polynomial features found from the original data. This can be seen as regularized regressions performed better with the polynomial features as opposed to the original features, as the polynomial regularized regressions yielded lower cross validated MAEs compared to non-polynomial regularized regressions. Additionally, neural networks and XGBoost regression also performed competitively but may be costly in terms of computational power required. All-in-all the results display the benefits of regularization and the use of polynomial features, as the utilization of both resulted in very competitive k-fold cross validated MAEs and outperformed all the other models evaluated.   
