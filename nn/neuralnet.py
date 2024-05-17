import numpy as np
from helpers import *

# a neural net with 1 hidden layer.
# the n_hidden here is for the number of neurons in that hidden layer
class NeuralNet:
    def __init__(self,
                 n_features,
                 n_hidden,
                 n_classes,
                 random_seed = 1024):

        super().__init__()

        self.n_classes = n_classes
        
        #initialize the random number generator
        rng = np.random.RandomState(random_seed)

        # Now, we initialize the weights and the bias in the hidden layer
        # Each neuron in the hidden layer is connected to each neuron
        # in the input layer. So, we need a matrix to represent the weights
        # hidden neuron1: weight w/input neuron1, weight w/input neuron2, ...
        # hidden neuron2: weight w/input neuron1, weight w/input neuron2, ...
        # hidden neuron3: weight w/input neuron1, weight w/input neuron2, ...

        # numpys random number generator will generate a randomized matrix
        # for us if we ask it to via the shape parameter to the normal() method
        # which is essentially a Gaussian distribution

        self.weights_h = rng.normal(loc=0.0, # mean center of the distribution
                                   scale=0.1, # spread/std dev of the distribution
                                   size=(n_hidden, n_features)) # shape of the matrix

        # the biases are initialized to 0s
        self.bias_h = np.zeros(n_hidden)

        # pretty much the same thing for the output layer of neurons
        self.weights_out = rng.normal(loc=0.0, # mean of the distribution
                                     scale=0.1, # spread or std dev
                                     size=(n_classes, n_hidden))

        self.bias_out = np.zeros(n_classes)


    # the forward pass, where we do the linear algebraic calculations
    # and apply the activation function
    # zh = input dot wh  + bh
    # ah = sigmoid(zh)
    # zout = ah dot wout + bout
    # aout = sigmoid(zout)
    def forward(self, x):

        # first for the hidden layer
        
        # input: n_examples x n_features
        # weight_h: n_hidden * n_features
        # for the dot product (i.e., matrix multiplication), we need to take the
        # transpose of the weight matrix
        # z_h.shape == n_examples x n_hidden
        # a_h.shape == n_examples x n_hidden
        z_h = np.dot(x, self.weights_h.T) + self.bias_h
        a_h = sigmoid(z_h)

        # next for the output layer
        # 
        # input: a_h with shape n_examples x n_hidden
        # weight_out: n_classes x n_hidden

        z_out = np.dot(a_h, self.weights_out.T) + self.bias_out
        a_out = sigmoid(z_out)

        return a_h, a_out
    
    def backward(self,
                 x,
                 a_h,
                 a_out,
                 y):

        # y is the intended class labels for the input feature data x,
        # which we will need to convert into
        # one hot encoding
        y_onehot = int_to_onehot(y, self.n_classes)
        
        # in the backward pass, all that we are trying to do is to calculate the
        # gradients, i.e., by how much we need to update the weights of the output
        # and hidden layer.

        # dLoss/dw_out = dLoss/da_out * da_out/dz_out * dz_out/dw_out
        # and we have already worked this out to be
        # dLoss/dw_out = 2*(a_out - y) * a_out(1 - a_out) * a_h

        # we calculate each derivative in the chain rule separately for ease
        # shape: n_examples x n_classes 
        d_loss__d_a_out = 2 * (a_out - y_onehot) / y.shape[0]

        # if you are wondering where the shape[0] thing came from, remember
        # that the loss for the entire set of output neurons is the MEAN squared error
        # Look at the paper notes for the derivation of the loss w.r.t. weights and bias

        # from sigmoid
        # shape: n_examples x n_classes
        d_a_out__d_z_out = a_out * (1 - a_out)

        # these first two derivatives, we will store their dot product
        # as delta_out for convenience
        # shape: n_examples x n_classes
        delta_out = d_loss__d_a_out * d_a_out__d_z_out

        # from linear equation
        # shape: n_examples x n_hidden
        d_z_out__d_w_out = a_h

        #d_loss__dw_out = d_loss__d_a_out * d_a_out__d_z_out * d_z_out__d_w_out
        # remember that the relationship between zout and wout is essentially a linear
        # equation, i.e, z = wa + b. The wa here is the dot product for a whole batch

        #shapes: [n_examples x n_classes] dot [n_examples x n_hidden]
        #therefore the transpose
        d_loss__d_w_out = np.dot(delta_out.T, d_z_out__d_w_out)
        d_loss__d_b_out = np.sum(delta_out, axis = 0)

        ##################################

        # Now, we need to find the gradients for the hidden layer's weights and biases
        # i.e., we need to figure out how the loss changes w.r.t the weights and biases
        # i.e., we need th derivative of the loss w.r.t weights and the biases

        # dLoss/dHiddenWeights = dLoss/d_a_out * d_a_out/d_z_out * d_z_out/d_a_h *
        #                        d_a_h/d_z_h * d_z_h/d_w_h

        # remember that dLoss/d_a_out * d_a_out/d_z_out == delta_out as computed above
        # shape of delta_out: n_examples x n_classes

        # see paper notes for the derivative
        # shape: n_classes x n_hidden
        d_z_out__d_a_h = self.weights_out

        # derivative of the loss wrt hidden layer activations
        # shape: n_examples x n_hidden
        d_loss__d_a_h = np.dot(delta_out, d_z_out__d_a_h)

        # see paper notes for the derivative of the sigmoid
        # shape: n_examples x n_hidden
        d_a_h__d_z_h = a_h * (1 - a_h)

        # see paper notes for the derivative
        # shape: n_examples x n_features
        d_z_h__d_w_h = x

        # input shapes:
        #   d_loss__d_a_h: n_examples x n_hidden
        #   d_a_h__d_z_h:  n_examples x n_hidden
        #   d_z_h__d_w_h:  n_examples x n_features
        # output shape: (basically the shape of the hidden layer)
        #   n_hidden x n_features
        d_loss__d_w_h = np.dot( (d_loss__d_a_h *  d_a_h__d_z_h).T,
                                      d_z_h__d_w_h)

        d_loss__d_b_h = np.sum((d_loss__d_a_h *  d_a_h__d_z_h), axis = 0)

        # return the gradients
        return (d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h, d_loss__d_b_h)


# the code
#nn = NeuralNet(3, 4, 2)                                     
# print(nn.weight_h)
# print(nn.weight_out)


