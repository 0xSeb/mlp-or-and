import numpy as np

andDataset = [
    {'input': [0, 0], 'expected_result': 0},
    {'input': [0, 1], 'expected_result': 0},
    {'input': [1, 0], 'expected_result': 0},
    {'input': [1, 1], 'expected_result': 1},
]

orDataset = [
    {'input': [0, 0], 'expected_result': 0},
    {'input': [0, 1], 'expected_result': 1},
    {'input': [1, 0], 'expected_result': 1},
    {'input': [1, 1], 'expected_result': 1},
]


def sigmoid(value):
    return np.tanh(value)


def derivative_sigmoid(value):
    return 1.0 - value ** 2


class MLP:
    def __init__(self, *args):
        self.numberOfLayers = len(args)
        self.shape = args
        self.layers = []
        self.weights = []

        self.init_layers()
        self.init_weights()

    def init_layers(self):
        for i in range(0, self.numberOfLayers):
            self.layers.append(np.ones(self.shape[i]))

    def print_layers(self):
        for i in range(0, len(self.layers)):
            print(self.layers[i])

    # Init weights between -0.25 and 0.25
    def init_weights(self):
        for i in range(self.numberOfLayers - 1):
            self.weights.append(np.zeros((self.layers[i].size,
                                          self.layers[i + 1].size)))

        for i in range(len(self.weights)):
            z = np.random.random((self.layers[i].size, self.layers[i + 1].size))
            self.weights[i][...] = (2 * z - 1) * 0.25

    def print_weights(self):
        for i in range(0, len(self.weights)):
            print(self.weights[i])

    def reset_weights(self):
        for i in range(len(self.weights)):
            z = np.random.random((self.layers[i].size, self.layers[i + 1].size))
            self.weights[i][...] = (2 * z - 1) * 0.25

    def propagate_forward(self, data):
        # Initialisation de la couche d'entrée
        for i in range(0, len(self.layers[0])):
            self.layers[0][i] = data["input"][i]
        # propagation avant de la couche précédente vers la couche suivante
        for i in range(1, self.numberOfLayers):
            self.layers[i][...] = sigmoid(np.dot(self.layers[i - 1], self.weights[i - 1]))
        # Return dernière couche pour pouvoir print pendant les tests
        return self.layers[-1]

    def propagate_backward(self, data, learning_rate=0.1):
        deltas = []

        # calcul de l'erreur sur la couche de sortie
        error = data["expected_result"] - self.layers[-1]
        # calcul de la différence de l'erreur sur la couche de sortie
        delta = error * derivative_sigmoid(self.layers[-1])
        deltas.append(delta)
        # On commence à l'index -2 car -1 est l'index de la couche de sortie
        for i in range(self.numberOfLayers - 2, 0, -1):
            # On calcule le delta en faisant
            # le produit de la dérivée de la somme des entrées et
            # le produit scalaire entre le delta et les poids de la couche précédente
            delta = derivative_sigmoid(self.layers[i]) * np.dot(deltas[0], self.weights[i].transpose())
            deltas.insert(0, delta)

        # Mise à jour des poids
        for i in range(len(self.weights)):
            # Étape permettant de modifier les matrices pour pouvoir faire le produit scalaire
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            delta_weigth = np.dot(layer.transpose(), delta)
            self.weights[i] += learning_rate * delta_weigth

        # Return error pour print pendant les tests
        return (error ** 2).sum()


if __name__ == '__main__':
    mlp = MLP(2, 10, 1)

    def learn(data, epochs, learning_rate):
        mlp.reset_weights()
        for i in range(epochs):
            n = np.random.randint(len(data))
            mlp.propagate_forward(data[n])
            mlp.propagate_backward(data[n], learning_rate)

        # Test
        for i in range(len(data)):
            o = mlp.propagate_forward(data[i])
            print(i, data[i]['input'], '%.2f' % o[0])
            print('(expected %.2f)' % data[i]['expected_result'], '\n')
            print('\n')


    learn(andDataset, 25000, 0.1)
    learn(orDataset, 25000, 0.1)
