import numpy as np


def sigmoid(x):
    """ Sigmoid like function using tanh """
    return np.tanh(x)


def dsigmoid(x):
    """ Derivative of sigmoid above """
    return 1.0 - x ** 2


class MLP:
    """ Multi-layer perceptron class. """

    def __init__(self, *args):
        """ Initialization of the perceptron with given sizes.  """
        self.shape = args
        n = len(args)

        # Build layers
        self.layers = []

        # Input layer
        self.layers.append(np.ones(self.shape[0]))
        # Hidden layer(s) + output layer
        for i in range(1, n):
            self.layers.append(np.ones(self.shape[i]))
        # Build weights matrix (randomly between -0.25 and +0.25)
        self.weights = []
        for i in range(n - 1):
            self.weights.append(np.zeros((self.layers[i].size,
                                          self.layers[i + 1].size)))

        # dw will hold last change in weights (for momentum)
        self.dw = [0, ] * len(self.weights)

        # Reset weights
        self.reset()

    def reset(self):
        """ Reset weights """

        for i in range(len(self.weights)):
            z = np.random.random((self.layers[i].size, self.layers[i + 1].size))
            self.weights[i][...] = (2 * z - 1) * 0.25

    def propagate_forward(self, data):
        """ Propagate data from input layer to output layer. """

        # Set input layer
        self.layers[0][0:] = data

        # Propagate from layer 0 to layer n-1 using sigmoid as activation function
        for i in range(1, len(self.shape)):
            # Propagate activity
            self.layers[i][...] = sigmoid(np.dot(self.layers[i - 1], self.weights[i - 1]))

        # Return output
        return self.layers[-1]

    def propagate_backward(self, target, lrate=0.1):
        """ Back propagate error related to target using lrate. """

        deltas = []

        # Compute error on output layer
        error = target - self.layers[-1]
        delta = error * dsigmoid(self.layers[-1])
        deltas.append(delta)

        # Compute error on hidden layers
        # -2 as -1 is the index of the last layer (output layer) which we do not want to compute
        for i in range(len(self.shape) - 2, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * dsigmoid(self.layers[i])
            deltas.insert(0, delta)

        # Update weights
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T, delta)
            self.weights[i] += lrate * dw + self.dw[i]
            self.dw[i] = dw

        # Return error
        return (error ** 2).sum()
