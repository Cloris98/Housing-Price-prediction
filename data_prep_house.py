import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler


class Preprocessing:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)

    # convert categorical variable into dummy
    def cate_dummy(self, data):
        data = pd.get_dummies(data)
        return data

    # def outliers(self, ):

    def split_data(self, X, Y):
        train_x, test_x, train_y, test_y = model_selection.train_test_split(X, Y, test_size=0.25, random_state=7)
        print()
        return train_x, test_x, train_y, test_y

    def process(self, test=False):
        """
        For Missing data:

            From analysing the training data, we found there are six features have more than 15% missing data. we should
        delete the corresponding variable and pretend it never existed. In this data set, they are 'PoolQC',
        'MiscFeature', 'Alley', 'Fence', 'FirePlaceQu', and 'LotFrontage'.
            Also, we can see that 'Garage-' variables have the same number of missing data, which means that the missing
        part refers to the same set of observations. Since the most important information regarding garages is
        expressed by 'GarageCars' and considering that we are just talking about 5% of missing data, so we will delete
        all the 'Garage_' variables. Same as 'Bsmt_' variables.
            Then, as we noticed from our heatmap, the 'MasVnrArea' and 'MasVnrType' variables are not significant, and
        they are have strong correlation with  'YearBuilt' and 'OverallQual' which we already considered, so I will
        delete these two variables.
        """
        drop_feature = ['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage', 'GarageYrBlt',
                        'GarageCond', 'GarageType', 'GarageFinish','GarageQual', 'BsmtFinType2', 'BsmtExposure',
                        'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'MasVnrArea', 'MasVnrType']
        if test:
            self.df = self.df.drop(drop_feature, axis=1)
            self.df = self.df.drop(self.df.loc[self.df['Electrical'].isnull()].index)
            self.df = self.cate_dummy(self.df)
            return self.df

        self.df = self.df.drop(drop_feature, axis=1)
        self.df = self.df.drop(self.df.loc[self.df['Electrical'].isnull()].index)
        self.df = self.cate_dummy(self.df)

        Y = self.df['SalePrice']
        X = self.df.drop('SalePrice', axis=1)

        return self.split_data(X, Y), X, Y




