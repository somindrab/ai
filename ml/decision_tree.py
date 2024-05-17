from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt

iris = datasets.load_iris()

# Here, I want to try using all available features (we've used only 2 of the 4 in other places)
X = iris.data[:, [0, 3]]
y = iris.target

# Next we are going to split the feature set and the class labels into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    stratify=y)

## Note that for decision trees we do not need to scale the feature set, i.e., no feature scaling needed

## Create the model and fit it
tree_model = DecisionTreeClassifier(criterion="gini",
                                    max_depth=3,
                                    random_state=1)

tree_model.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined, y_combined, classifier=tree_model)

plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.tight_layout()

#plt.savefig('images/02_08.png', dpi=300)
plt.show()


## actually show the decision tree
from sklearn import tree
feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
tree.plot_tree(tree_model, feature_names=feature_names, filled=True)
plt.show()
