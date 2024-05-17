from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt

iris = datasets.load_iris()

# Here, I want to try using all available features (we've used only 2 of the 4 in other places)
X = iris.data[:, [0, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    stratify=y)

# For K Nearest Neighbors classification, since distance matters, the features have to be standardized.
sc = StandardScaler()
sc.fit(X_train)
## What's happening with fit() is that this calculates the sample mean (mu or x bar) and the standard deviation (sigma). Remember that the formula is
## x' = (x - xbar) / sigma

# transforms applies that formula with the calculated values to the dataset
X_train_std = sc.transform(X_train)

# Now that these are calculated by the StandardScaler, we will transform the test dataset with the same values - we won't call fit() on the test dataset
# That would destroy the purpose here since we would not be testing with the same parameters!
X_test_std = sc.transform(X_test)

knn = KNeighborsClassifier(n_neighbors = 5,
                           p=2,
                           metric='minkowski')

#See the book for an explanation of p. Its way too involved to type here, and Som, just see the formula for minkowski distance
#that'll make it clear

knn.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined, classifier=knn, test_idx=range(105,150))

plt.xlabel('Petal length [cm] - standardized')
plt.ylabel('Petal width [cm] - standardized')
plt.legend(loc='upper left')
plt.tight_layout()

#plt.savefig('images/02_08.png', dpi=300)
plt.show()


