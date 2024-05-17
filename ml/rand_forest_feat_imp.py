from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

feature_labels = df_wine.columns[1:]

X, y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=0,
                                                    stratify=y)

forest = RandomForestClassifier(n_estimators=500, random_state=1)

forest.fit(X_train, y_train)

importances = forest.feature_importances_
#print(importances)

#argsort will return the *indices* of the data which is sorted in ascending order
#[::-1] will reverse those indices, so the index at 0 will be the index of the greatest object (float in this case)
indices = np.argsort(importances)[::-1]
#print(indices)

for f in range(X_train.shape[1]):
    print(feature_labels[indices[f]], importances[indices[f]])

    
