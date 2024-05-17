import numpy as np

### sigmoid activation function
## sigmoid(z) = 1/(1 + e^-z)
def sigmoid(z):
    z = np.float128(z)
    return 1. / (1. + np.exp(-z))

#leaky
def relu(z):
#    print(z)
    r = np.float128(np.maximum(0.01*z,z))
#    print(r)
    return r

#leaky
def differenciate_relu(x):
    dx = np.copy(x)
    dx[dx>0.01] = 1.
    dx[dx<=0.01] = 0.01
    # print('Differenciation Incoming z: \n')
    # print(x)
    # print('Differenciation Outgoing dz: \n')
    # print(dx)
    return dx

## we usually think about one hot encoding as a vector with one bit hot
## but we need to think about training datasets now, and for that, we need
## to understand that we need to represent the one hot encoded vectors for
## the entire batch or dataset
## so the one hot encoding will need to represent
## number of examples x number of classes
## for training example 0: 0 0 0 1 0 0 0 0
## for training example 1: 0 1 0 0 0 0 0 0
## for training example 2: 0 0 0 0 0 1 0 0
## ....
## ....
## ....

def int_to_onehot(y, num_labels):
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1
    return ary

# this returns batches of features and class labels based on the size of the batch
# this implements a co-routine sort of a function which
# keeps state and can be called multiple times
# also, this isn't just a slicer of data - it shuffles the input and
# returns a batch of shuffled data
def minibatch_generator(X, y, minibatch_size):
    indices = np.arange(X.shape[0])

    np.random.shuffle(indices)

    for start_idx in range(0,
                           indices.shape[0] - minibatch_size + 1,
                           minibatch_size):

        batch_idx = indices[start_idx:start_idx + minibatch_size]

        yield X[batch_idx], y[batch_idx]


# Loss calculation. Loss here is Mean Squared Error (MSE)
# loss is the mean of (target - probability (i.e. prediction))^2
# here target is simply a list of the correct class labels in a list
# they correspond to the input examples
def mse_loss(targets, probas, num_labels=10):

    #but first, we need to convert our targets into one_hot format
    onehot_targets = int_to_onehot(targets, num_labels=num_labels)

    return np.mean( (onehot_targets - probas) ** 2)

# this is a measure of how accurate the model is
def accuracy(targets, predicted_labels):
    return np.mean(predicted_labels == targets)
    

def compute_mse_and_acc(model, X, y, num_labels=10, minibatch_size=100):

    correct_pred = 0
    mse = 0.
    num_examples = 0
    
    minibatch_gen = minibatch_generator(X, y, minibatch_size)

    for i, (features, targets) in enumerate(minibatch_gen):

        _, outputs = model.forward(features)

        loss = mse_loss(targets, outputs, num_labels)

        predicted_labels = np.argmax(outputs, axis = 1)

        correct_pred += (predicted_labels == targets).sum()

        num_examples += targets.shape[0]

        mse += loss

    mse = mse/i
    acc = correct_pred/num_examples

    return mse,acc

        
def train(model, X_train, y_train, X_valid, y_valid,
          num_epochs, learning_rate=0.1):

    epoch_train_acc = []
    epoch_valid_acc = [] 
    epoch_loss = [] 

    
    for e in range(num_epochs):

        minibatch = minibatch_generator(X_train, y_train, minibatch_size=100)

        for X_train_mini, y_train_mini in minibatch:

            # get the predictions from the forward pass
            a_h, a_out = model.forward(X_train_mini)

            # get the gradients from the backward pass
            d_loss__d_weights_out, d_loss__d_bias_out, \
            d_loss__d_weights_h, d_loss__d_bias_h \
            = model.backward(X_train_mini, a_h, a_out, y_train_mini)

            # update the weights
            model.weights_out -= learning_rate * d_loss__d_weights_out
            model.bias_out -= learning_rate * d_loss__d_bias_out

            model.weights_h -= learning_rate * d_loss__d_weights_h
            model.bias_h -= learning_rate * d_loss__d_bias_h

        ## Now that we have completed one adjustment, we are going to
        ## record the loss and accuracy of the model with the entire training
        ## and validation data for this epoch. Note that this is just a forward
        ## pass, and does not adjust the weights anymore in this epoch

        train_mse, train_acc = compute_mse_and_acc(model, X_train, y_train)

        valid_mse, valid_acc = compute_mse_and_acc(model, X_valid, y_valid)

        # convert into percentages
        train_acc, valid_acc = train_acc*100, valid_acc*100

        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        epoch_loss.append(train_mse)

        print(f'Epoch {e+1}/{num_epochs} | Train MSE: {train_mse} | Train Acc: {train_acc} | Valid Acc: {valid_acc}')

    return epoch_train_acc, epoch_valid_acc, epoch_loss


def dnn_compute_mse_and_acc(model, X, y, num_labels=10, minibatch_size=100):

    correct_pred = 0
    mse = 0.
    num_examples = 0
    
    minibatch_gen = minibatch_generator(X, y, minibatch_size)

    for i, (features, targets) in enumerate(minibatch_gen):

        outputs = model.forward(features)

        loss = mse_loss(targets, outputs, num_labels)

        predicted_labels = np.argmax(outputs, axis = 1)

        correct_pred += (predicted_labels == targets).sum()

        num_examples += targets.shape[0]

        mse += loss

    mse = mse/i
    acc = correct_pred/num_examples

    return mse,acc


def dnn_train(model, X_train, y_train, X_valid, y_valid,
          num_epochs, learning_rate=0.1):

    epoch_train_acc = []
    epoch_valid_acc = [] 
    epoch_loss = [] 

    
    for e in range(num_epochs):

        minibatch = minibatch_generator(X_train, y_train, minibatch_size=100)

        for X_train_mini, y_train_mini in minibatch:

            # get the predictions from the forward pass
            predictions = model.forward(X_train_mini)

            # get the gradients from the backward pass
            model.backward(y_train_mini, predictions, learning_rate)

        ## Now that we have completed one adjustment, we are going to
        ## record the loss and accuracy of the model with the entire training
        ## and validation data for this epoch. Note that this is just a forward
        ## pass, and does not adjust the weights anymore in this epoch

        train_mse, train_acc = dnn_compute_mse_and_acc(model, X_train, y_train)

        valid_mse, valid_acc = dnn_compute_mse_and_acc(model, X_valid, y_valid)

        # convert into percentages
        train_acc, valid_acc = train_acc*100, valid_acc*100

        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        epoch_loss.append(train_mse)

        print(f'Epoch {e+1}/{num_epochs} | Train MSE: {train_mse} | Train Acc: {train_acc} | Valid Acc: {valid_acc}')

    return epoch_train_acc, epoch_valid_acc, epoch_loss
