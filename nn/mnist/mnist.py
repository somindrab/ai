from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import sys
sys.path.append('/home/som/prog/pyml/nn')
from helpers import *
from neuralnet import NeuralNet


X, y = fetch_openml('mnist_784',
                    version=1,
                    return_X_y=True)

# fetch_openml will return pandas DataFrames
# .values will return the underlying numpy arrays
X = X.values
y = y.astype(int).values

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
#     break

# print(X_train_mini.shape)
# print(y_train_mini.shape)

model = NeuralNet(n_features=28*28,
                  n_hidden=50,
                  n_classes=10)

# _, outputs = model.forward(X_valid)

# mse = mse_loss(y_valid, outputs, num_labels=10)

# #argmax selects the index of the largets number
# predicted_labels = np.argmax(outputs, axis=1)
# acc = accuracy(y_valid, predicted_labels)

# print(mse)
# print(acc)

# mse, acc = compute_mse_and_acc(model, X_valid, y_valid)

# print(mse)
# print(acc)

train(model, X_train, y_train, X_valid, y_valid, num_epochs=50, learning_rate=0.1)
