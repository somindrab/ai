from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from deepneuralnet_relu import DeepNeuralNet
from deepneuralnet_relu import InputLayer
from deepneuralnet_relu import Layer

#load the DeepNeuralNet Model stored in file
file = sys.argv[1]
model = pickle.load(open(file, "rb"))

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

X_test_subset = X_test[:1000, :]
y_test_subset = y_test[:1000]

predictions = model.forward(X_test_subset)

test_pred = np.argmax(predictions, axis=1)

print(f'Accuracy: {np.count_nonzero( (y_test_subset==test_pred) == True) * 100 / X_test_subset.shape[0]}%')

misclassified_images = X_test_subset[y_test_subset != test_pred][:25]

misclassified_labels = test_pred[y_test_subset != test_pred][:25]

correct_labels = y_test_subset[y_test_subset != test_pred][:25]

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True, figsize=(8,8))

ax = ax.flatten()

for i in range(25):
    img = (misclassified_images[i].reshape(28,28))*255
    ax[i].imshow(img, cmap='Greys',interpolation='nearest')
    ax[i].set_title(f'{i+1}) '
                    f'True: {correct_labels[i]}\n'
                    f'Predicted: {misclassified_labels[i]}')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
