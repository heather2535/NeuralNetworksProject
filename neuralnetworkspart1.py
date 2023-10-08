import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    #layer initialization
     def __init__(self, n_inputs, n_neurons):
         #initialize wieghts and biases
          self.weights = np.random.randn(n_inputs, n_neurons)
          self.biases = np.zeros((1, n_neurons))
    # Forward pass
     def forward(self, inputs):
         #calculate values from inputs, weights and biases
          self.output = np.dot(inputs, self.weights) + self.biases

#Rectolinear activation
class Activation_ReLU:
    #forward pass
    def forward(self,inputs):
        #calculates output values from inputs
        self.output = np.maximum(0, inputs)

# softmax activation
class Activation_Softmax:
    def forward(self, inputs):
        # calculate unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))  # exponentiation
        # normalize for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# Calculate loss class
class Loss:
    # calculate the data and regularization losses
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_pred.shape) == 1:
            correct_confidences == y_pred_clipped[
                range(samples), y_true]



X,y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense2= Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

