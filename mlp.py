import numpy as np
import matplotlib.pyplot as plt

class MLP():
    def __init__(self): #initialize params with dimensions of training data and no. neurons hidden layer
        self.W1 = np.random.rand(10, 784) - 0.5 #generate values between 0 and 1 for elements in array dim1Xdim2
        self.b1 = np.random.rand(10, 1) -0.5
        self.W2 = np.random.rand(24, 10) -0.5 
        self.b2 = np.random.rand(24, 1) -0.5
        self.W3 = np.random.rand(24, 24) -0.5
        self.b3 = np.random.rand(24, 1) -0.5
    
    def get_params(self): #returns weights and biases
        return self.W1, self.b1, self.W2, self.b2, self.W3, self.b3

    def ReLU(self, x): #standard relu function
        return np.maximum(0, x)
    
    def deriv_ReLU(self, Z): #derivative of the relu function
        return Z > 0
    
    def softmax(self, Z): #standard softmax function
        A = np.exp(Z) / sum(np.exp(Z))
        return A
    
    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y


    def forward_pass(self, X): #forward pass through the neural net

        Z1 = self.W1.dot(X) + self.b1 #dot product with weights matrix
        A1 = self.ReLU(Z1) #activation function
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self.ReLU(Z2)
        Z3 = self.W3.dot(A2) + self.b3
        A3 = self.softmax(Z3)

        return Z1, A1, Z2, A2, Z3, A3

    def backward_prop(self, Z1, A1, Z2, A2, Z3, A3, X, Y): #compute gradients
        m = Y.size
        one_hot_Y = self.one_hot(Y) #one hot encode labels

        dZ3 = A3 - one_hot_Y
        dW3 = 1 / m * dZ3.dot(A2.T)
        db3 = 1 / m * np.sum(dZ3)

        dZ2 = self.W3.T.dot(dZ3) * self.deriv_ReLU(Z2)
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2)

        dZ1 = self.W2.T.dot(dZ2) * self.deriv_ReLU(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ2)

        return dW1, db1, dW2, db2, dW3, db3
    
    def update_params(self, dW1, db1, dW2, db2, dW3, db3, alpha): #update parameters through gradient descent
        self.W1 = self.W1 - alpha * dW1
        self.b1 = self.b1 - alpha * db1    
        self.W2 = self.W2 - alpha * dW2  
        self.b2 = self.b2 - alpha * db2
        self.W3 = self.W3 - alpha * dW3  
        self.b3 = self.b3 - alpha * db3    
    
    def get_predictions(self, A3): #select index of max value after softmax
        return np.argmax(A3, 0)

    def get_accuracy(self, predictions, Y): #compare predictions with true labels
        return np.sum(predictions == Y) / Y.size

    def gradient_descent(self, X_train, Y_train, X_val, Y_val, alpha, iterations):
        self.iter_arr = np.arange(iterations)
        self.accuracies = []
        self.val_accuracies = []
        for i in range(iterations): #for n iterations update parameters
            Z1, A1, Z2, A2, Z3, A3 = self.forward_pass(X_train)
            dW1, db1, dW2, db2, dW3, db3 = self.backward_prop(Z1, A1, Z2, A2, Z3, A3, X_train, Y_train)
            self.update_params(dW1, db1, dW2, db2, dW3, db3, alpha)
            
            predictions = self.get_predictions(A3)
            acc = self.get_accuracy(predictions, Y_train)
            self.accuracies.append(acc)

            val_predictions = self.make_predictions(X_val)
            val_acc = self.get_accuracy(val_predictions, Y_val)
            self.val_accuracies.append(val_acc)

            if i % 1000 == 0:
                print("Iteration no. ", i)
                print("Train acc: ", acc)
                print("Val acc: ", val_acc)
    
    def train(self, X_train, y_train, X_val, y_val, alpha, iterations): #call gradient descent func
        self.gradient_descent(X_train, y_train, X_val, y_val, alpha, iterations)
    
    def learning_curve(self): #plot accuracy over iterations
        plt.plot(self.iter_arr, self.accuracies, label = 'training accuracy')
        plt.plot(self.iter_arr, self.val_accuracies, label = 'validation accuracy')
        plt.xlabel('iterations')
        plt.ylabel('accuracy')

        plt.show
    
    def make_predictions(self, X): #make predictions for new input data
        _, _, _, _, _, A3 = self.forward_pass(X)
        predictions = self.get_predictions(A3)
        return predictions
    
    def test(self, X_test, y_test):
        predictions = self.make_predictions(X_test)
        test_acc = self.get_accuracy(predictions, y_test)
        print("Test Accuracy: ", test_acc)
        

