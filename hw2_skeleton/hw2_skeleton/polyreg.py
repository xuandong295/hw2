'''
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np


#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self, degree = 1, regLambda = 1E-8):
        '''
        Constructor
        '''
        #TODO
        self.degree = degree
        self.regLambda = regLambda

    def polyfeatures(self, X, degree):
        '''
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not inlude the zero-th power.

        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        '''
        #TODO
        # X.shape = (n, 1)
        # A.shape = (n, d)     d: degree ()
        '''
            X = [[1],
                [2],
                [3]]
            degree = 3
            A = [[1, 1, 1],
                 [2, 4, 8],
                 [3, 9, 27]]
        '''
        A = X
        if degree > 1:
            for i in range(2, degree+1):
                A = np.hstack((A, X**1))
        return A

        
        

    def fit(self, X, y):
        '''
            Trains the model
            Arguments:
                X is a n-by-1 array
                y is an n-by-1 array
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling
                at first
        '''
        #TODO
        n = len(X)
        X = self.polyfeatures(X, self.degree)

        self.standardizationValues(X)
        X = self.standardize(X)

        # Add 1 to beginning of X
        X = np.c_[np.ones([n, 1]), X];

        n,d = X.shape

        # construct reg matrix
        regMatrix = self.regLambda * np.eye(d)
        regMatrix[0,0] = 0

        self.theta = np.linalg.pinv(X.T.dot(X) + regMatrix).dot(X.T).dot(y);

        print("Calculated theta: ")
        print(self.theta)
        
    def standardizationValues(self, X):
        n, d = X.shape
        result = np.ones([n, d])
        self.transformation = np.zeros([d, 2])

        for i in range(0, d):
            feature = np.empty(n)
            for j in range(0, n):
                feature[j] = X[j, i]
            mean = np.mean(feature)
            stdev = np.std(feature)

            self.transformation[i, 0] = mean
            self.transformation[i, 1] = stdev

    def standardize(self, X):

        n, d = X.shape

        result = np.zeros([n, d])

        for i in range(0, n):

            for j in range(0, d):
                if self.transformation[j, 1] == 0:
                    result[i, j] = 0
                else:
                    result[i, j] = (X[i, j] - self.transformation[j, 0])/self.transformation[j, 1]

        return result   
        
    def predict(self, X):
        '''
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        '''
        # TODO
        n = len(X)

        X = self.polyfeatures(X, self.degree)
        X = self.standardize(X)
        
        # add 1s column
        X = np.c_[np.ones([n, 1]), X]

        

        return X.dot(self.theta)



#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------


def learningCurve(Xtrain, Ytrain, Xtest, Ytest, regLambda, degree):
    '''
    Compute learning curve
        
    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        regLambda -- regularization factor
        degree -- polynomial degree
        
    Returns:
        errorTrains -- errorTrains[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTests -- errorTrains[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]
        
    Note:
        errorTrains[0:1] and errorTests[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    '''
    
    n = len(Xtrain)
    
    errorTrain = np.zeros((n))
    errorTest = np.zeros((n))
    for i in range(2, n):
        Xtrain_subset = Xtrain[:(i+1)]
        Ytrain_subset = Ytrain[:(i+1)]
        model = PolynomialRegression(degree, regLambda)
        model.fit(Xtrain_subset,Ytrain_subset)
        
        predictTrain = model.predict(Xtrain_subset)
        err = predictTrain - Ytrain_subset
        errorTrain[i] = np.multiply(err, err).mean()
        
        predictTest = model.predict(Xtest)
        err = predictTest - Ytest
        errorTest[i] = np.multiply(err, err).mean()
    
    return (errorTrain, errorTest)