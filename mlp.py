import numpy as np

class MLP():
    def __init__(self): #initialize params with dimensions of training data and no. neurons hidden layer
        self.W1 = np.random.rand(10, 784) #generate values between 0 and 1 for elements in array dim1Xdim2
        self.b1 = np.random.rand(10, 1)
        self.W2 = np.random.rand(24, 10)
        self.b2 = np.random.rand(24, 1)
        self.W3 = np.random.rand(24, 24)
        self.b3 = np.random.rand(24, 1)
    
    def get_params(self):
        return self.W1, self.b1, self.W2, self.b2, self.W3, self.b3

    def ReLU(self, x): #standard relu function
        return np.maximum(0, x)
    
    def deriv_ReLU(Z):
        return Z > 0
    
    def softmax(self, x): #standard softmax function
        return np.exp(x) / np.sum(np.exp(x))
    
    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def forward_pass(self, X):
        Z1 = self.W1.dot(X) + self.b1 #dot product with weights matrix
        A1 = self.ReLU(Z1) #activation function
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self.ReLU(Z2)
        Z3 = self.W3.dot(A2) + self.b3
        A3 = self.softmax(Z3)

        return Z1, A1, Z2, A2, Z3, A3

    def backward_prop(self, Z1, A1, Z2, A2, Z3, A3, X, Y):
        m = Y.size
        one_hot_Y = self.one_hot(Y) #one hot encode labels

        dZ3 = A3 - one_hot_Y
        dW3 = 1 / m * dZ3.dot(A2.T)
        db3 = 1 / m * np.sum(dZ3, 2)

        dZ2 = self.W3.T.dot(dZ3) * self.deriv_ReLU(Z2)
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2, 2)

        dZ1 = self.W2.T.dot(dZ2) * self.deriv_ReLU(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ2, 2)

        return dW1, db1, dW2, db2, dW3, db3
    
    def update_params(self, dW1, db1, dW2, db2, dW3, db3, alpha):
        self.W1 = self.W1 - alpha * dW1
        self.b1 = self.b1 - alpha * db1    
        self.W2 = self.W2 - alpha * dW2  
        self.b2 = self.b2 - alpha * db2
        self.W3 = self.W3 - alpha * dW3  
        self.b3 = self.b3 - alpha * db3    
    
    def get_predictions(self, A2):
        return np.argmax(A2, 0)

    def get_accuracy(self, predictions, Y):
        print(predictions, Y)
        return np.sum(predictions == Y) / Y.size

    def gradient_descent(self, X, Y, alpha, iterations):
        for i in range(iterations):
            Z1, A1, Z2, A2, Z3, A3 = self.forward_pass(X)
            dW1, db1, dW2, db2 = self.backward_prop(Z1, A1, Z2, A2, Z3, A3, X, Y)
            self.update_params(dW1, db1, dW2, db2, alpha)
            if i % 10 == 0:
                print("Iteration: ", i)
                predictions = self.get_predictions(A2)
                print(self.get_accuracy(predictions, Y))
    
    def make_predictions(self, X):
        _, _, _, _, _, A3 = self.forward_pass(X)
        predictions = self.get_predictions(A3)
        return predictions

