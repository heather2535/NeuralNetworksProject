import math
import numpy as np
import nnfs



# input --> exponentiation (remove negative values without losing meaning) --> normalization (probability dist)
# softmax activation = exponentiation and normalization
nnfs.init()

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)  #exponentiation
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True) #normalization

print(norm_values)
