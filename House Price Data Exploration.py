# -*- coding: utf-8 -*-
"""Kaggle House Price.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_VbsWEZW_bikJBEuLBvTAfLucVt2KEbu

# Kaggle House Price Data Exploration

## import data from drive
"""

!pip install -U -q PyDrive

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# TRAINING DATA: https://drive.google.com/file/d/1rrWkFLniD4WSWtaqyWk1MBwHSgurgYjj/view?usp=sharing
# TEST DATA: https://drive.google.com/file/d/1k1L2abHWKzldS-4G-IyPBloUwWncuFl5/view?usp=sharing
id_train = '1rrWkFLniD4WSWtaqyWk1MBwHSgurgYjj'
file_train = drive.CreateFile({'id': id_train})
file_train.GetContentFile('train.csv')

id_test = '1k1L2abHWKzldS-4G-IyPBloUwWncuFl5'
file_test = drive.CreateFile({'id': id_test})
file_test.GetContentFile('test.csv')

# importing packages
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df.head()
# test_df.head()

"""## Data Exploration

#### Understand the Raw Data
"""

house_price_df = pd.read_csv('train.csv')

house_price_df.head()

# check data information
house_price_df.info()

# check the unique values for each column

house_price_df.nunique()

"""### analysis the target variable - 'SalePrice'"""

# get target variable
y = house_price_df['SalePrice']

y.describe()

sns.distplot(y)

print('Skewness: %f' % y.skew())
print('Kurtosis: %f' % y.kurt())

"""#### Understand the features"""

# check missing values
house_price_df.isnull().sum()

# Understand Numerical feature
# By analysing the data_description file, we find the followng features are numerical 
num_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
            'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF',
            'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
house_price_df[num_cols].describe()



# Numerical feature
# Grlivearea with saleprice
var = 'GrLivArea'
data = pd.concat([house_price_df['SalePrice'], house_price_df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

# totalbsmesf with saleprice
var = 'TotalBsmtSF'
data = pd.concat([house_price_df['SalePrice'], house_price_df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

# Category feature
_,axss = plt.subplots(2,2, figsize=[20,10])
sns.boxplot(x = 'SalePrice', y = 'Street', data = house_price_df, ax=axss[0][0])
sns.boxplot(x = 'SalePrice', y = 'Heating', data = house_price_df, ax=axss[0][1])
sns.boxplot(x = 'SalePrice', y = 'BsmtQual', data = house_price_df, ax=axss[1][0])
sns.boxplot(x = 'SalePrice', y = 'SaleCondition', data = house_price_df, ax=axss[1][1])

# relationship between yearbuilt and saleprice
var = 'YearBuilt'
data = pd.concat([house_price_df['SalePrice'], house_price_df[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);

# correlation matrix 
corrmat = house_price_df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

"""Heatmap analysis: 

> Two squares show a significant correlation, which is the 'TotalBasmtSF' and '1stFlrSF' variables and the 'GarageCars' and 'GarageArea' variables. This strong correlation indicates a situation of multicollinearity, which means they give almost the same information.  
>From 'SalePrice' correlation, we can see some well-know variables such as 'OverallQual', 'GrLivArea' and 'TotalBsmtSF'. But there are still some variables we should take into account. 


"""

# SalePrice correlation matrix
# top 10 correlation variables
cols = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index
coeff = np.corrcoef(house_price_df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(coeff, cbar=True, annot=True,  fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(house_price_df[cols], size = 2.5)
plt.show();

"""## Missing Data"""

total = house_price_df.isnull().sum().sort_values(ascending=False)
percentage = ((house_price_df.isnull().sum()) / house_price_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percentage], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

missing_data[missing_data['Total'] > 0].index

# numerical feature
drop_feature = ['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage', 'GarageYrBlt', 'GarageCond', 'GarageType', 'GarageFinish', 'GarageQual', 'BsmtFinType2', 'BsmtExposure', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'MasVnrArea', 'MasVnrType']

house_price_df = pd.get_dummies(house_price_df)

from sklearn import model_selection
train_x, test_x, train_y, test_y = model_selection.train_test_split(house_price_df, y, test_size=0.25, random_state=1)
train_x.head()

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
n_folds = 5
scorer = make_scorer(mean_squared_error,greater_is_better = False)
def rmse_CV_train(model):
    kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(house_price_df.values)
    rmse = np.sqrt(-cross_val_score(model,train_x, train_y,scoring ="neg_mean_squared_error",cv=kf))
    return (rmse)
def rmse_CV_test(model):
    kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(house_price_df.values)
    rmse = np.sqrt(-cross_val_score(model,test_x, test_y,scoring ="neg_mean_squared_error",cv=kf))
    return (rmse)

lr = LinearRegression()
lr.fit(train_x, train_y)
test_pre = lr.predict(test_x)
train_pre = lr.predict(train_x)
print(test_pre, train_pre)




