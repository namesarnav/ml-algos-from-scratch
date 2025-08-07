
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LassoRegression: 
    
    '''
    Linear Regression Model that also combines feature selecion 
    Equation = (SSE) + lambda * sum(abs(w[j]))
    w = weights of jth iteration
    '''
    
    def __init__(self, learning_rate, l1_penalty, iterations) -> None:
        '''
        l1 penalty --> lambda term in the equations
        learning rate
        iterations --> n in the equation
        '''
        self.lr = learning_rate
        self.l1_penalty = l1_penalty
        self.iterations = iterations
        
    
    def fit(self, X, Y): 
        '''
        The method that is used to train the model. Initialize with weights `w` and bias `b`
        '''
        
        self.X = X, 
        self.Y = Y
        
        # m is number of training examples
        # n is number of features
        self.m, self.n = X.shape 
        
        # we use numpy's method to initalize weights as zeros
        self.W = np.zeros(self.n)
        
        # Bias will always be init as 0
        self.b = 0 
        
        for i in range(self.iterations): 
            self.update_weights()
       
        return self
    
    def update_weights(self): 
        
        Y_pred = self.predict(self.X)
        
        
        # dW is gradient of each feature adjusted for L1 regularization term
        dW = np.zeros(self.n)
        
        for j in range(self.n):
            if self.W[j] > 0:
                dW[j] = (-2 * (self.X).dot(self.Y - Y_pred) + self.l1_penalty) / self.m
            else:
                dW[j] = (-2 * (self.X).dot(self.Y - Y_pred) - self.l1_penalty) / self.m
        # db calculates the gradient of the bias term
        
        db = -2 * np.sum(self.Y - Y_pred) / self.m
        self.W = self.W - self.lr * dW
        self.b = self.b - self.lr * db
   
        return self
   
    def predict(self, X): 
        return X.dot(self.W) + self.b
        