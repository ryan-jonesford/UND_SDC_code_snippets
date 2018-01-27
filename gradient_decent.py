"""
Credit- Udacity Self Driving Car Nanodegree Lesson 4.30
"""
import numpy as np

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1/(1+np.exp(-x))

learnrate = 0.5
x = np.array([1, 2])
y = np.array(0.5)

# Initial weights
w = np.array([0.5, -0.5])

# Calculate one gradient descent step for each weight
# Calculate output of neural network (aka y_hat)
y_hat = sigmoid(x[0]*w[0] + x[1]*w[1])
nn_output = y_hat

# Calculate error of neural network
error = (y - y_hat)

# Calculate change in weights
del_w = [learnrate * error * y_hat * (1 - y_hat) * x[0],learnrate * error * y_hat * (1 - y_hat) * x[1]]

print('Neural Network output:')
print(y_hat)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(del_w)