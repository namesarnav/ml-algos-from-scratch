
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
        
        
        
def main():
    df = pd.read_csv("Experience-Salary.csv")
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, 1].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=1/3, random_state=0)

    model = LassoRegression(
        iterations=1000, learning_rate=0.01, l1_penalty=500)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    print("Predicted values: ", np.round(Y_pred[:3], 2))
    print("Real values:      ", Y_test[:3])
    print("Trained W:        ", round(model.W[0], 2))
    print("Trained b:        ", round(model.b, 2))

    plt.scatter(X_test, Y_test, color='blue', label='Actual Data')
    plt.plot(X_test, Y_pred, color='orange', label='Lasso Regression Line')
    plt.title('Salary vs Experience (Lasso Regression)')
    plt.xlabel('Years of Experience (Standardized)')
    plt.ylabel('Salary')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()