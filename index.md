# Midterm Project 
## Introduction 
  The goal of this project is to evaluate the performance of different regressors when predicting the price of a property using the features within the boston housing data set, as well as polynomial features created from the original data. First, I explored the use of regularized regression on polynomial features created from the original data to predct housing prices in boston. In particular I usied six different regularization techniques: LASSO, Ridge, Elastic Net, SCAD (Smoothly-Clipped Absolute Deviation), Square Root Lasso, and Stepwise. Further I explored the use kernel regression, Random Forest, XGBoost, and Neural networks, and ultimately compared all models to a baseline linear regression model. I used K-fold cross validation to examine each model's mean absolute error. The model that yields the lowest mean absolute error when predicting the price would be the most accurate. 
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
#### Implementation 
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

### Implementation
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
Code for 
## 

