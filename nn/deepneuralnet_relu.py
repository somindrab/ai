from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from helpers import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

class Layer:

    def __init__(self,
                 layer_number,
                 n_neurons_self,
                 n_neurons_input,
                 prevHiddenLayer,
                 random_seed=1024):

        super().__init__()

        #removing the random seed here so that each layer will have unique weights
        self.rng = np.random.RandomState()

        self.layer_number = layer_number
        
        self.n_neurons_self = n_neurons_self

        self.n_neurons_input = n_neurons_input

        self.prevHiddenLayer = prevHiddenLayer

        self.weights = np.float128(self.rng.normal(loc=0.0,
                                                   scale=0.1,
                                                   size=(self.n_neurons_self, self.n_neurons_input)))

 #       print(f'Number of 0 weights: {np.count_nonzero(self.weights == 0)}')

        self.bias = np.float128(np.zeros(n_neurons_self))

        self.weights_bias_initialized = False

    def forward(self,prev_a):

        #print(f'Forward pass for Layer: {self.layer_number} \n')

        #thislayer_z = W . a_prev + bias
        #thislayer_a = sigmoid(thislayer_z)

        #input shape: [n_examples, n_features] or [n_examples, n_hidden]
        #hidden layer shape: [n_hidden, n_features]
        #output shape after transposing hidden layer: [n_examples, n_hidden]

        #print(f'prev_a: \n{prev_a}')
        #print(f'self.weights: \n{self.weights}')
        #print(f'self.bias: \n{self.bias}')
        self.z = np.float128(np.dot(prev_a, self.weights.T) + self.bias)

        #self.a = sigmoid(self.z)
        self.a = np.float128(relu(self.z))
        #print(self.a)
#        print(f'Number of 0 activations: {np.count_nonzero(self.a == 0)}')
        
        return self.a

    def backward(self, layer_to_the_right, d_loss__d_prev_z, prev_a, learning_rate):

        #print(f'Backward pass for layer: {self.layer_number}')
        #d_loss__d_prev_z
        #shape: (nexamples, nhidden)
        
        #shape: (n_hidden, n_hidden) or (examples, classes) if output layer
        d_prev_z__d_a = layer_to_the_right.weights

        #shape: (n_examples, n_hidden)
        #d_a__d_z = self.a * (1. - self.a)
        d_a__d_z = differenciate_relu(self.z)
        

        #shape: (n_examples, n_hidden)
        d_z__d_w = prev_a

        d_loss__d_a = np.dot(d_loss__d_prev_z, d_prev_z__d_a)
        
        d_loss__d_w = np.dot( (d_loss__d_a * d_a__d_z).T, d_z__d_w)

        d_loss__d_b = np.sum( (d_loss__d_a * d_a__d_z), axis=0)
        
        #Adjust the weights and the bias
        self.updateWeightsAndBias(d_loss__d_w, d_loss__d_b, learning_rate)

        d_loss__d_z = d_loss__d_a * d_a__d_z #the derivative of the loss to this layer's z, which is the starting point for backprop's next backward()

        return d_loss__d_z

    def updateWeightsAndBias(self, d_loss__d_w, d_loss__d_b, learning_rate):
        self.weights -= learning_rate * d_loss__d_w
        self.bias -= learning_rate * d_loss__d_b
        
        

class InputLayer:

    def __init__(self,
                 n_neurons_self):

        super().__init__()

        self.n_neurons_self = n_neurons_self

        self.a = None

# A neural net with mulitple hidden layers
# I <-> H1 <-> H2 <-> .... <-> HN <-> O
class DeepNeuralNet:

    def __init__(self,
                 n_features,
                 n_classes,
                 n_hidden,
                 n_hlayers,
                 random_seed=1024):

        self.n_classes = n_classes
        
        #create an input layer based on the number of features
        self.inputLayer = InputLayer(n_features)

        self.hidden_layers = []

        self.n_hlayers = n_hlayers

        # the first hidden and last hidden layer are different since they connect to the
        # input and output layers, which (most likely) have a different number of neurons than the hidden layers

        # first hidden layer. Connects to the input layer
        layer = Layer(0, # the first hidden layer
                      n_hidden,
                      n_features,
                      prevHiddenLayer=self.inputLayer,
                      random_seed=1024)

        self.hidden_layers.append(layer)

        for l in range(n_hlayers - 1):
            layer = Layer(l+1, # input layer is 0, first hidden layer is 1. Both constructed above
                          n_hidden,
                          n_hidden,
                          prevHiddenLayer=layer,
                          random_seed=1024)

            self.hidden_layers.append(layer)

        self.outputLayer = Layer(1024, #just a placeholder layer number for the output layer
                                 n_classes,
                                 n_hidden,
                                 prevHiddenLayer=layer,
                                 random_seed=1024)
                      
   
    def forward(self, x):

        #print(f'Forward pass:')

        # z = w * aprev + b
        # a = sigmoid(z)

        #the first interconnections between the input layer and the first hidden layer

        #input shape: [n_examples, n_features]
        #1st hidden layer shape: [n_hidden, n_features]
        #output shape after transposing hidden layer: [n_examples, n_hidden]

        # the activations for the input layer is basically just the training data applied to the input layer
        # shape: (n_examples, n_features)
        self.inputLayer.a = prev_a = np.float128(x)
        

        for i,layer in enumerate(self.hidden_layers):
            prev_a = layer.forward(prev_a) #prev_a's shape for hidden layers is (n_examples, n_hidden)


        #finally, applied at the output layer, we get our output activations
        output_a = self.outputLayer.forward(prev_a)

        return output_a

    def backward(self, y, a_out, learning_rate):

        #print(f'Backward pass for output layer')
        # start from the output layer
        # convert the class labels (y) into onehot encoding
        y_onehot = int_to_onehot(y, self.n_classes)

        d_loss__d_a_out = 2. * (a_out - y_onehot) / y.shape[0]
        #d_a_out__d_z_out = a_out * (1. - a_out)
        d_a_out__d_z_out = differenciate_relu(self.outputLayer.z)
        
        #the derivative of the loss to z of the output layer
        d_loss__d_z_out = d_loss__d_a_out * d_a_out__d_z_out
        d_z_out__d_w_out = self.outputLayer.prevHiddenLayer.a

        d_loss__d_w_out = np.dot(d_loss__d_z_out.T, d_z_out__d_w_out)
        d_loss__d_b_out = np.sum(d_loss__d_z_out, axis=0)

        self.outputLayer.updateWeightsAndBias(d_loss__d_w_out, d_loss__d_b_out, learning_rate)
        #print(f'Finished backward pass for output layer')        

        # for the hidden layers
        d_loss__d_prev_z = d_loss__d_z_out
        layer_to_the_right = self.outputLayer
        for l in reversed(self.hidden_layers):
            d_loss__d_prev_z = l.backward(layer_to_the_right, d_loss__d_prev_z, l.prevHiddenLayer.a, learning_rate)
            layer_to_the_right = l
        

########################################

# X, y = fetch_openml('mnist_784',
#                     version=1,
#                     return_X_y=True)

# # fetch_openml will return pandas DataFrames
# # .values will return the underlying numpy arrays
# X = X.values
# y = y.astype(int).values

# ## there are a total of 70,000 examples with 70000 labels.
# ## we want to use 55000 for training, 5000 for validation, and 10000 for testing
# X_temp, X_test, y_temp, y_test = train_test_split(X,
#                                                   y,
#                                                   test_size=10000,
#                                                   random_state=123,
#                                                   stratify=y)

# X_train, X_valid, y_train, y_valid = train_test_split(X_temp,
#                                                       y_temp,
#                                                       test_size=5000,
#                                                       random_state=123,
#                                                       stratify=y_temp)


# # minibatch_gen = minibatch_generator(X_train, y_train, 100)

# # for X_train_mini, y_train_mini in minibatch_gen:
# #      break

# # print(X_train_mini.shape)
# # print(y_train_mini.shape)

# model = DeepNeuralNet(n_features=28*28,
#                       n_classes=10,
#                       n_hidden=50,
#                       n_hlayers=1)

# dnn_train(model, X_train, y_train, X_valid, y_valid, num_epochs=30, learning_rate=0.0001)

# # dump the model to a file so that we can use it later
# file = sys.argv[1]
# pickle.dump(model, open(file, "wb"))

# outputs = model.forward(X_valid)

# # print(outputs)

# mse = mse_loss(y_valid, outputs, num_labels=10)

# # # #argmax selects the index of the largets number
# predicted_labels = np.argmax(outputs, axis=1)
# acc = accuracy(y_valid, predicted_labels)

# print(mse)
# print(acc)

# X_test_subset = X_test[:1000, :]
# y_test_subset = y_test[:1000]

# predictions = model.forward(X_test_subset)

# test_pred = np.argmax(predictions, axis=1)

# misclassified_images = X_test_subset[y_test_subset != test_pred][:25]

# misclassified_labels = test_pred[y_test_subset != test_pred][:25]

# correct_labels = y_test_subset[y_test_subset != test_pred][:25]

# fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True, figsize=(8,8))

# ax = ax.flatten()

# for i in range(25):
#     img = misclassified_images[i].reshape(28,28)
#     ax[i].imshow(img, cmap='Greys',interpolation='nearest')
#     ax[i].set_title(f'{i+1}) '
#                     f'True: {correct_labels[i]}\n'
#                     f'Predicted: {misclassified_labels[i]}')

# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# plt.show()
