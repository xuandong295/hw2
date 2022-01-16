'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np

class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, epsilon=0.0001, maxNumIters = 10000):
        '''
        Constructor
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
    

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
        '''
        n, d = X.shape

        arg1 = 0

        for i in range(0, n):

            xVector = np.matrix(X[i]).T

            arg1 += -y[i]*np.log(self.sigmoid(theta.dot(xVector))) - (1 - y[i])*np.log(1 - self.sigmoid(theta.dot(xVector)))

        arg2 = 0

        for i in range(0, d):
            arg2 +=  theta[i] ** 2

        arg2 = (regLambda/2) * arg2

        final = arg1 + arg2

        return final.item(0, 0)
    
    
    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        '''
        n, d = X.shape

        gradient = np.empty(d)

        #first find first element of gradient

        result = 0
        for i in range(0, n):
            xVector = np.matrix(X[i]).T
            result += self.sigmoid(theta.dot(xVector)) - y[i]

        gradient[0] = result

        for j in range(1, d):
            result = 0

            for i in range(0, n):
                xVector = np.matrix(X[i]).T
                result += (self.sigmoid(theta.dot(xVector)) - y[i])*X[i, j]

            result = result + regLambda * theta[j]
            gradient[j] = result

        return gradient   


    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''
        n = len(X)
        # Must add 1 to beginning of X
        X = np.c_[np.ones([n, 1]), X]

        n,d = X.shape
        d = d-1  # remove 1 for the extra column of ones we added to get the original num features

        self.theta = np.zeros(d + 1)

        self.JHist =[]

        iters = 0

        self.JHist.append((self.computeCost(self.theta, X, y, self.regLambda), self.theta))
        self.theta = self.theta - (self.alpha * self.computeGradient(self.theta, X, y, self.regLambda))

        while iters <= self.maxNumIters and not \
            (self.hasConverged(self.theta, self.JHist[len(self.JHist) - 1][1], self.epsilon)):


            self.JHist.append((self.computeCost(self.theta, X, y, self.regLambda), self.theta))

            adjustment = self.alpha * self.computeGradient(self.theta, X, y, self.regLambda)

            self.theta = self.theta + adjustment

            iters += 1

        self.JHist.append((self.computeCost(self.theta, X, y, self.regLambda), self.theta))

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        n = len(X)
        
        # add 1s column
        X = np.c_[np.ones([n, 1]), X]

        # predict
        result = X.dot(self.theta)

        for i in range(0, len(result)):
            if result[i] >= .5:
                result[i] = 0
            else:
                result[i] = 1

        return result


    def sigmoid(self, Z):
        '''
        Computes the sigmoid function 1/(1+exp(-z))
        '''
        dimensions = Z.shape

        if len(dimensions) == 1:
            n = dimensions[0]
            result = np.empty(n)
            for i in range(0, n):
                result[i] = 1/(1 + np.exp(Z[i]))
            return result

        n, d = Z.shape
        result = np.empty([n, d])

        for i in range(0, n):
            for j in range(0, d):

                result[i, j] = 1/(1 + np.exp(Z[i, j]))

        return result

    def hasConverged(self, new, old, epsilon):
        total = 0
        for i in range(0, len(new)):
            total += (new[i] - old[i]) ** 2
        return total <= epsilon
