import pandas as pd
from io import StringIO


csv_data = \
'''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
print(df)

#print(df.isnull().sum())

from sklearn.impute import SimpleImputer
import numpy as np

imr = SimpleImputer(missing_values=np.nan,
                    strategy='mean')

imr.fit(df.values)
imputed_data = imr.transform(df.values)
#print(imputed_data)

imputed_df = df.fillna(df.mean)
print(imputed_df)
print(type(imputed_df))
