import matplotlib.pyplot as plt


class ModelEval:
    def __init__(self, lr, ridgecv, rf, test_x, test_y):
        self.lr = lr
        self.ridge = ridgecv
        self.rf = rf
        self.test_x = test_x
        self.test_y = test_y

    def linear_reg_plot(self):
        plt.scatter(self.test_x['GrLivArea'], self.lr, c='red',  label='predict value')
        plt.scatter(self.test_x['GrLivArea'], self.test_y, c='green',  label='actual value')
        plt.title('Linear Regression')
        # plt.xlabel('predicted values')
        plt.legend(loc='upper left')
        plt.ylabel('Price')

    # def plot_residuals(self, train_y, test_y):
    #     plt.scatter(self.train_pre, self.train_pre - train_y, c='blue', label='Training Data')
    #     plt.scatter(self.test_pre, self.test_pre - test_y, c='green', marker='v', label='Validation Data')
    #     # plt.title('Linear Regression')
    #     plt.xlabel('Predicted Values')
    #     plt.ylabel('Residuals')
    #     plt.legend(loc='upper left')
    #     plt.hlines(y=0, xmin=0, xmax=500000, color='yellow')

    def linear_reg_with_ridge(self):
        plt.scatter(self.test_x['GrLivArea'], self.ridge, c="red", label='predict value')
        plt.scatter(self.test_x['GrLivArea'], self.test_y, c="green", label='actual value')
        plt.title("Linear regression with Ridge regularization")
        plt.legend(loc='upper left')
        # plt.xlabel("Predicted values")
        # plt.ylabel("Real values")

    # def plot_residuals_ridge(self, y):
    #     plt.scatter(self.train_y_rdg, self.train_y_rdg - train_y, c='blue', label='Training Data')
    #     plt.scatter(self.test_y_rdg, self.test_y_rdg - test_y, c='black', marker='v', label='Validation Data')
    #     # plt.title('Linear Regression with Ridge Regularization')
    #     plt.xlabel('Predicted Values')
    #     plt.ylabel('Residuals')
    #     plt.legend(loc='upper left')
    #     plt.hlines(y=0, xmin=0, xmax=600000, color='red')

    def rf_(self):
        plt.scatter(self.test_x['GrLivArea'], self.rf, c="red", label='predict value')
        plt.scatter(self.test_x['GrLivArea'], self.test_y, c="green", label='actual value')
        plt.title("Random Forest")
        plt.legend(loc='upper left')
        # plt.xlabel("Predicted values")
        plt.ylabel("Price")


    def eval(self):
        plt.subplot(2, 2, 1)
        self.linear_reg_plot()
        plt.subplot(2, 2, 2)
        self.linear_reg_with_ridge()
        plt.subplot(2, 2, 3)
        self.rf_()
        # plt.subplot(2, 2, 4)
        # self.plot_residuals_ridge(train_y, test_y)
        plt.show()

    def whole_(self):
        plt.scatter(self.test_x['GrLivArea'], self.lr, c='red', s=15, label='linear regression')
        plt.scatter(self.test_x['GrLivArea'], self.rf, c="orange", s=15, label='random forest')
        plt.scatter(self.test_x['GrLivArea'], self.ridge, c="lightseagreen", s=15,  label='LR with ridge')
        plt.scatter(self.test_x['GrLivArea'], self.test_y, c='slategray', s=15, label='actual value')
        plt.title('Linear Regression')
        # plt.xlabel('predicted values')
        plt.legend(loc='upper left')
        plt.ylabel('Price')
        plt.show()




