import pandas as pd
from pandas import DataFrame
import numpy as np

from sklearn.model_selection import train_test_split

df_wine = pd.read_csv('/home/som/prog/pyml/Datasets/wine.data',
                      header=None)

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

#print("Class Labels: ", np.unique(df_wine['Class label']))

#print(df_wine.head())

#print(df_wine.columns[0:])

X, y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=0,
                                                    stratify=y)

# shape is a tuple representing the shape. the [1] indexes into the tuple, yielding the number of columns.
# in this case that's 14 columns: 1 class label + 13 features
dim = X.shape[1]
#print("dim: ", dim)
#print(X.shape)
#print(type(X))

indices = tuple([1,3,5])
print(indices)

print(X_train[:,indices])

#print(indices)

from itertools import combinations

total = 0
for p in combinations(indices, r=dim-1):
    total +=1

#print(total)

# remember that stratify keeps the same proportion of class labels in the training and test datasets as the original dataset
