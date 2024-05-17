from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

## Remember, no feature scaling needed for decision trees and therefore none for Random Forests either

forest = RandomForestClassifier(n_estimators = 25,
                                random_state = 1,
                                n_jobs = 2)

#n_estimators is the number of decision trees in the forest
#n_jobs is how many cores to use to parallelize the operation

forest.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined, y_combined, classifier=forest)

plt.xlabel('Petal length [cm]')
plt.ylabel('Petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()

#plt.savefig('images/02_08.png', dpi=300)
plt.show()
                            
                                
