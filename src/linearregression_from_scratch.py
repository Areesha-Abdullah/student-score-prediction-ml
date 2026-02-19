import numpy as np
class LinearRegressionScratch:
    def __init__(self):
        self.coeff_ = None
        self.intercept_ = None

    def fit(self, X_train , y_train):

        X_train = np.array(X_train, dtype=float)
        y_train = np.array(y_train, dtype=float)

        #to add column of 1 in x matrix to make it n*(m+1
        X_train =np.insert(X_train, 0 , 1 , axis=1)

        # calculate the coeffcients by formula
        # find matrix B = ((X^T * X)*-1)* (X*T)*Y
        # where X is the matrix containing all the input features of every element + 1 and  Y is matrix of all the target values of every element 

        betas = np.linalg.inv(np.dot(X_train.T , X_train )).dot(X_train.T).dot(y_train)
        print(betas)
        self.intercept_ = betas[0]
        self.coeff_ = betas[1:]


        

    def predict(self, X_test):
        # to find y_pred y_pred=B0 + (B(coefficient) . Xtest)
        y_pred = self.intercept_ + np.dot(X_test, self.coeff_)
        print ( y_pred )
        return y_pred
        