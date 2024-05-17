import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import torch
from torch.nn.functional import one_hot
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

# the goals here are:
# 0. Predicting the fuel efficiency (mpg) of cars
# 1. import the dataset from the source. We have downloaded it on disk instead of fetching remotely every single time because we cheap.
# 2. Pre-process the data so that it is trainable
  # 2.1. Normalize the continuous features. That means (feature_value - mean)/stddev
  # 2.2. Bucketize the model year
  # 2.3. Either encode or embed the origin. We are going to use one-hot encoding, i.e. we will encode


# import the dataset as a DataFrame
# we have the dataset downloaded from https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data in prog/pyml/pytorch/Datasets

    # 1. mpg:           continuous
    # 2. cylinders:     multi-valued discrete
    # 3. displacement:  continuous
    # 4. horsepower:    continuous
    # 5. weight:        continuous
    # 6. acceleration:  continuous
    # 7. model year:    multi-valued discrete
    # 8. origin:        multi-valued discrete
    # 9. car name:      string (unique for each instance)


DATASET_LOC = "/home/som/prog/pyml/pytorch/Datasets/auto-mpg.data"

column_names = ["mpg", "cylinders", "displacement", "horsepower",
                "weight", "acceleration", "model year", "origin"]

#create the dataframe from the data in the file and the names of the columns from auto-mpg.names file
#note that the .data file does not have the column names, so we must supply them to while creating the DataFrame

df = pd.read_csv(DATASET_LOC,
                 names=column_names,
                 na_values="?",
                 comment='\t',
                 sep=" ",
                 skipinitialspace=True)

df = df.dropna()

#this is something that we are doing and not done in the book
#this is because when normalizing the "cyclinders", which is int64,
#it complains that we are turning an int dtyped column to something that is not int without an explicit cast
#so we set the dtype of cylinders to float64.
#note that setting the entire dataframe as float is a bad idea
#because we want to bucketize and encode the model year and origin, and they are best
#served as ints
df = df.astype({'cylinders':"float64"})
#print(df.dtypes)

df = df.reset_index(drop=True)

df_train, df_test = train_test_split(df,
                                     train_size=0.8,
                                     random_state=1)

train_stats = df_train.describe().transpose()

#if you are wondering what the describe() method does, here is the output transpose()d:

# print(train_stats)

#               count         mean         std     min     25%     50%     75%     max
# mpg           313.0    23.404153    7.666909     9.0    17.5    23.0    29.0    46.6
# cylinders     313.0     5.402556    1.701506     3.0     4.0     4.0     8.0     8.0
# displacement  313.0   189.512780  102.675646    68.0   104.0   140.0   260.0   455.0
# horsepower    313.0   102.929712   37.919046    46.0    75.0    92.0   120.0   230.0
# weight        313.0  2961.198083  848.602146  1613.0  2219.0  2755.0  3574.0  5140.0
# acceleration  313.0    15.704473    2.725399     8.5    14.0    15.5    17.3    24.8
# model year    313.0    75.929712    3.675305    70.0    73.0    76.0    79.0    82.0
# origin        313.0     1.591054    0.807923     1.0     1.0     1.0     2.0     3.0

#and we basically need the mean and the std to be able to normalize.
#see how easy this is in python
#we are going to normalize the numeric values and so we need those column names
#model year and origin are ordinal (ordered categorical) and nominal (unordered categorical)

numeric_columns=["cylinders", "displacement", "horsepower",
                "weight", "acceleration"]

#and now we do the normalizing on the copies of this data
df_train_norm, df_test_norm = df_train.copy(), df_test.copy()

for col in numeric_columns:
    mean = train_stats.loc[col,'mean']
    std = train_stats.loc[col, 'std']
    df_train_norm.loc[:,col] = (df_train_norm.loc[:,col] - mean)/std
    df_test_norm.loc[:,col] = (df_test_norm.loc[:,col]  - mean)/std

#now let's bucketize the model year column
#we will choose 4 buckets
#bucket 0: year < 73
#bucket 1: 73 <= year < 76
#bucket 2: 76 <= year < 79
#bucket 3: year >= 79

boundaries = torch.tensor([73, 76, 79])
v = torch.tensor(df_train_norm['model year'].values)
#see https://pytorch.org/docs/stable/generated/torch.bucketize.html
df_train_norm['model year bucketed'] = torch.bucketize(v,
                                                       boundaries,
                                                       right=True)

v = torch.tensor(df_test_norm['model year'].values)
df_test_norm['model year bucketed'] = torch.bucketize(v,
                                                      boundaries,
                                                      right=True)
numeric_columns.append("model year bucketed")

#now let's onehot encode the origin
#values are 1,2,3

norigins = len(set(df_train_norm['origin']))
#one_hot takes a tensor and returns a tensor
#see https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html
onehot_encoded = one_hot(torch.from_numpy(df_train_norm['origin'].values) % norigins)
x_train_numeric = torch.tensor(df_train_norm[numeric_columns].values)
X_train = torch.cat([x_train_numeric, onehot_encoded], 1).float()

## NOTE: the thing about the .cat of onehot_encoded is that it adds 3 columns
## to x_train_numeric - that's how the one hot encoding is represented in the training data
## so that's 6 feature columns + 3 new columns = 9 features

onehot_encoded = one_hot(torch.from_numpy(df_test_norm['origin'].values) % norigins)
x_test_numeric = torch.tensor(df_test_norm[numeric_columns].values)
X_test = torch.cat([x_test_numeric, onehot_encoded], 1).float()

#create the label tensors
y_train = torch.tensor(df_train_norm['mpg'].values).float()
y_test = torch.tensor(df_test_norm['mpg'].values).float()

#create a DataLoader from the training data
train_ds = TensorDataset(X_train, y_train)
torch.manual_seed(1)
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)

#create a model
#(Input: 9 features) applied to Hidden(8) <-> Hidden(4) <-> Output(1 class)

model = nn.Sequential(nn.Linear(9,8),
                      nn.ReLU(),
                      nn.Linear(8,4),
                      nn.ReLU(),
                      nn.Linear(4,1))

#loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

torch.manual_seed(1)
nepochs = 200
train_loss = 0
for epoch in range(nepochs):
    for x_batch, y_batch in train_dl:
        #model(x_batch) returns a tensor of shape [8,1]. For loss_fn we need shape [8]
        pred = model(x_batch)[:,0]
        loss=loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
    print(f'Epoch {epoch+1} | Loss: {train_loss/len(train_dl)}')
    train_loss = 0

#https://pytorch.org/docs/stable/generated/torch.no_grad.html
#Context manager that disables gradient calculations
#uses less memory, speeds stuff up
#very useful when we know we want to infer, such as now
#where we want to test with y_test
with torch.no_grad():
    pred = model(X_test)[:,0]
    loss = loss_fn(pred, y_test)
    print(f'Test Mean Loss Squared Error (MSE): {loss.item()}')
    print(f'Test Mean Absolute Error (MAE): {nn.L1Loss()(pred, y_test)}')

