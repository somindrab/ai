from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np
from plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt


# Load the iris dataset
iris = datasets.load_iris()

# We are interested in the two specific features. We will only use those out of the 4 features avaialble
X = iris.data[:, [2, 3]]
y = iris.target

# Next we are going to split the feature set and the class labels into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    stratify=y)

# Next, we are going to scale our features. We will use Standardization (see Wikipedia or handwritten notes). This is also called Z-score normalization
# Very important for gradient descent to be optimal.
sc = StandardScaler()
sc.fit(X_train)
## What's happening with fit() is that this calculates the sample mean (mu or x bar) and the standard deviation (sigma). Remember that the formula is
## x' = (x - xbar) / sigma

# transforms applies that formula with the calculated values to the dataset
X_train_std = sc.transform(X_train)

# Now that these are calculated by the StandardScaler, we will transform the test dataset with the same values - we won't call fit() on the test dataset
# That would destroy the purpose here since we would not be testing with the same parameters!
X_test_std = sc.transform(X_test)

# Now, use the perceptron to create the model
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

#"Show features and corresponding labels. Ask it to create a model co-relating the features to the model"

# Use this to make predictions on the test dataset and see how many matched vs did not match
y_pred = ppn.predict(X_test_std)

print("Total misclassified: " + str((y_test != y_pred).sum()))
print("Total size of test dataset: " + str(y_test.size))

# We can use the metrics module to figure out the accuracy score
print(accuracy_score(y_test, y_pred))

plot_decision_regions(X_train, y_train, ppn)
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')


#plt.savefig('images/02_08.png', dpi=300)
plt.show()
