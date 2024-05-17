from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from helpers import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from deepneuralnet_relu import DeepNeuralNet
from deepneuralnet_relu import InputLayer
from deepneuralnet_relu import Layer

X, y = fetch_openml('mnist_784',
                    version=1,
                    return_X_y=True)

# fetch_openml will return pandas DataFrames
# .values will return the underlying numpy arrays
X_u = X.values
y = y.astype(int).values

X = np.float128(X_u/255)

## there are a total of 70,000 examples with 70000 labels.
## we want to use 55000 for training, 5000 for validation, and 10000 for testing
X_temp, X_test, y_temp, y_test = train_test_split(X,
                                                  y,
                                                  test_size=10000,
                                                  random_state=123,
                                                  stratify=y)

X_train, X_valid, y_train, y_valid = train_test_split(X_temp,
                                                      y_temp,
                                                      test_size=5000,
                                                      random_state=123,
                                                      stratify=y_temp)


# minibatch_gen = minibatch_generator(X_train, y_train, 100)

# for X_train_mini, y_train_mini in minibatch_gen:
#      break

# print(X_train_mini.shape)
# print(y_train_mini.shape)

model = DeepNeuralNet(n_features=28*28,
                      n_classes=10,
                      n_hidden=15,
                      n_hlayers=4)

dnn_train(model, X_train, y_train, X_valid, y_valid, num_epochs=100, learning_rate=0.016)

# dump the model to a file so that we can use it later
file = sys.argv[1]
pickle.dump(model, open(file, "wb"))
