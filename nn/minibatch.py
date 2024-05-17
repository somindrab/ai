import numpy as np

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



