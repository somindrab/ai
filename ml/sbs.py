from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#So, you want to implement a Sequential Backward Selection Algorithm

#General steps:
#Create a model with all feature columns. Score it.
#Remove a feature column. Run the model. Score it
#Keep doing this until you get to the desired number of columns.
#Keep track of the scores. The best score will be the subset you want.

class SBS:
    def __init__(self, estimator, accuracy_scorer=accuracy_score, random_state=1, k_features=1):
        self.estimator_ = clone(estimator)
        self.accuracy_scorer_ = accuracy_scorer
        self.random_state_ = random_state
        self.k_features_ = k_features

    # this is simply going to call an estimator (or classifier or model or whatever we want to call it)
    # on a training set with the features we pass it to and then score
    # this is essentially scoring the estimator who we will select (like KNN) on the subset of features
    # that we pass it to the function
    def _calculate_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator_.fit(X_train[:,indices]  , y_train)
        y_pred = self.estimator_.predict(X_test[:, indices])
        score = self.accuracy_scorer_(y_test, y_pred)
        return score


    def fit(self, X, y):

        # let's first score with the entire feature space

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.3,
                                                            random_state=self.random_state_)
                                                            #stratify=y)#
        
        # this is essentially the number of columns in the dataset
        #shape of the training dataset is (178,14).
        #14 since the first column is the class label, rest of them are 13 features
        #Remember that X is not the dataset, it is the feature information of training dataset and y is the class labels of the training dataset
        #so X_train has 13 columns, y_train has 1
        dim = X_train.shape[1]

        #We can index into the X_train with a tuple of column indices.
        #Tuple is essentially an object that holds a list
        #tuple(list[1,3,5]) = a tuple of 1,3,5
        indices=tuple(range(dim))
        
        score = self._calculate_score(X_train, y_train, X_test, y_test, indices)
    
        #Remember, what we have to keep track of here is the score for a subset of columns (given by the column indices)
        #a list of tuples
        self.column_subsets_ = [indices]

        #a list of scores
        self.scores_ = [score]

        while (dim > self.k_features_):
            scores = []
            subsets = []

            #for a given set of number of features, we are going to pick every possible combination
            #of columns, and we are going to score them

            for p in combinations(indices, r=dim - 1):

                score = self._calculate_score(X_train, y_train, X_test, y_test, p)

                scores.append(score)
                subsets.append(p)


            #what was the best score?
            best = np.argmax(scores)
            #which subset of features did that belong to? Store that subset and correspondingly
            #its score
            self.column_subsets_.append(subsets[best])
            self.scores_.append(scores[best])

            dim -= 1


        return self



#Use the wine dataset and KNN to find if we can select a set of features that won't hurt the accuracy too much if
#we were to drop them. What would be a good set?

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df_wine = pd.read_csv('/home/som/prog/pyml/Datasets/wine.data',
                      header=None)

X, y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=0,
                                                    stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

knn = KNeighborsClassifier(n_neighbors = 5)
sbs = SBS(knn, k_features=1)

sbs.fit(X_train_std, y_train)

k_feat = [len(k) for k in sbs.column_subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()

# look at Plots/sbs.png. With least number of features with the max score (1.0) is 3. Let's see what those are
# (Pdb) print(sbs.column_subsets_)
#[(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12), (0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12), (0, 1, 2, 3, 4, 6, 7, 10, 11, 12), (0, 1, 2, 3, 4, 6, 9, 10, 12), (0, 1, 2, 3, 4, 6, 9, 12), (0, 1, 4, 6, 8, 10, 12), (0, 1, 4, 6, 10, 12), (0, 1, 6, 10, 12), (0, 6, 10, 12), (2, 6, 12), (6, 9), (6,)]

# the tuple with 3 features is indexed 10.

k3 = list(sbs.column_subsets_[10])

df_wine.columns = ['Class label',
                   'Alcohol',
                   'Malic acid',
                   'Ash',
                   'Alcalinity of ash',
                   'Magnesium',
                   'Total phenols',
                   'Flavonoids',
                   'Nonflavonoid phenols',
                   'Proanthocyanins',
                   'Color intensity',
                   'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']

print(df_wine.columns[1:][k3])
