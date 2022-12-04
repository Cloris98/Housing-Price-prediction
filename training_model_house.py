import pandas as pd
import statistics as stat
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold


class TrainModel:
    def __init__(self, data):
        self.train_x = data[0][0]
        self.test_x = data[0][1]
        self.train_y = data[0][2]
        self.test_y = data[0][3]
        self.train = data[1]

    def linear_reg(self):
        lr = LinearRegression()
        lr.fit(self.train_x, self.train_y)
        test_pred = lr.predict(self.test_x)
        # train_pred = lr.predict(self.train_x)
        print('rmse on train', stat.median(self.rmse_cross_val_train(lr)))
        print('rmse on Test', stat.median(self.rmse_cross_val_test(lr)))
        return test_pred

    def rmse_cross_val_train(self, model):
        n_folds = 5
        kf = KFold(n_folds, shuffle=True, random_state=0).get_n_splits(self.train.values)
        rmse = np.sqrt(-cross_val_score(model, self.train_x, self.train_y, scoring='neg_mean_squared_error', cv=kf))
        return rmse

    def rmse_cross_val_test(self, model):
        kf = KFold(5, shuffle=True, random_state=0).get_n_splits(self.train.values)
        rmse = np.sqrt(-cross_val_score(model, self.test_x, self.test_y, scoring='neg_mean_squared_error', cv=kf))
        return rmse

    def RF_param(self):
        parameters = {
            'n_estimators': [60, 80, 100],
            'max_depth': [10, 15, 20, 25, 30],
            'max_features': [10, 15, 20, 25, 30]
        }
        Grid_RF = GridSearchCV(RandomForestRegressor(), parameters, cv=5)
        Grid_RF.fit(self.train_x, self.train_y)
        print(Grid_RF.best_params_)

    def rf_model(self):
        rf_best_model = RandomForestRegressor(random_state=0, max_features=25, n_estimators=80, max_depth=25)
        rf_best_model.fit(self.train_x, self.train_y)
        rf_pred = rf_best_model.predict(self.test_x)
        print('Random Forest RMSE on Training Set: ', stat.median(self.rmse_cross_val_train(rf_best_model)))
        print('Random Forest RMSE on Testing Set:', stat.median(self.rmse_cross_val_test(rf_best_model)))
        return rf_pred

    def ridge_cv(self):
        ridge = RidgeCV(alphas=[0.01, 0.03, 0.05, 0.06, 0.1, 0.3, 1, 5, 10, 15, 30])
        ridge.fit(self.train_x, self.train_y)
        alpha = ridge.alpha_
        # try for more precision with alphas centered around above 'alpha'
        ridge_centered = RidgeCV(alphas=[alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85,
                                alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15, alpha * 1.25,
                                alpha * 1.3, alpha * 1.35, alpha * 1.4], cv=5)
        ridge_centered.fit(self.train_x, self.train_y)
        alpha_centered = ridge_centered.alpha_
        print('best alpha: ', alpha_centered)
        print('Ridge RMSE on Training Set: ', stat.median(self.rmse_cross_val_train(ridge_centered)))
        print('Ridge RMSE on Testing Set: ', stat.median(self.rmse_cross_val_test(ridge_centered)))
        # y_train_ridge = ridge_centered.predict(self.train_x)
        y_test_ridge = ridge_centered.predict(self.test_x)
        return y_test_ridge

    def linear_model(self):
        lr = self.linear_reg()
        lr_with_ridge = self.ridge_cv()
        # rf = self.RF_param()
        rf = self.rf_model()

        return lr, lr_with_ridge, rf

