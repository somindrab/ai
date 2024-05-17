from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn

iris = load_iris()
X = iris['data']
y = iris['target']

# here, I am taking a deviation from the book based on my own experience
# with creating DeepNeuralNet in ../nn
# let's normalize the feature set off the bat, instead of normalizing
# the training set before training and the test set before testing

X = (X - np.mean(X)) / np.std(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1./3, random_state=1)

# convert into a pytorch tensor from the numpy array
X_train = torch.from_numpy(X_train).float()

y_train = torch.from_numpy(y_train)

train_ds = TensorDataset(X_train, y_train)

torch.manual_seed(1)

batch_size=2

train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# we are going to create a class to represent a Neural Network
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        #two hidden layers. output considered hidden just like we did too
        #in our own design :-)
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self,x):
        x = self.layer1(x)
        x = nn.Sigmoid()(x)
        x = self.layer2(x)
        x = nn.Softmax()(x)
        return x

    

#instantiate the model
model = Model(X.shape[1], #num of neurons in the input layer==num of features
              16, #hidden layer of 16 neurons
              3) #3 class labels == size of the output layer
              
learning_rate = 0.01

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), learning_rate)

num_epochs = 100
loss_hist = [0]*num_epochs
acc_hist = [0]*num_epochs

for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        #send the batch through the model. forward pass
        pred = model(x_batch)
        #calculate the loss based on prediction and the actual class labels
        loss = loss_fn(pred, y_batch)
        #back prop
        loss.backward()
        #adjust weights
        optimizer.step()
        optimizer.zero_grad()
        loss_hist[epoch] += loss.item() * y_batch.size(0)
        is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
        acc_hist[epoch] += is_correct.sum()

    loss_hist[epoch] /= len(train_dl.dataset)
    acc_hist[epoch] /= len(train_dl.dataset)

    print(f'Epoch {epoch+1} | Loss: {loss_hist[epoch]} | Accuracy: {acc_hist[epoch]}')

    
#### test with the test dataset
X_test = torch.from_numpy(X_test).float()
pred_test = model(X_test)
y_test = torch.from_numpy(y_test)

y_predicted_argmaxed = torch.argmax(pred_test, dim=1)
for i in range(y_test.shape[0]):
    print(f'Predicted: {y_predicted_argmaxed[i]}, Real: {y_test[i]}')

is_correct = (torch.argmax(pred_test, dim=1) == y_test).float()
acc_test = is_correct.mean()

print(f'Test accuracy: {acc_test * 100}%')
