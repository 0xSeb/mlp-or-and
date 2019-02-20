import numpy as np
import mlp

if __name__ == '__main__':

    network = mlp.MLP(2, 20, 20, 1)
    samples = np.zeros(4, dtype=[('input', float, 2), ('output', float, 1)])


    def learn(network, samples, epochs=10000, lrate=0.0001):
        # Train
        for i in range(epochs):
            n = np.random.randint(samples.size)
            network.propagate_forward(samples['input'][n])
            network.propagate_backward(samples['output'][n], lrate)
        # Test
        for i in range(samples.size):
            o = network.propagate_forward(samples['input'][i])
            print(i, samples['input'][i], '%.2f' % o[0])
            print('(expected %.2f)' % samples['output'][i], '\n')
            print('\n')


    # Example 1 : OR logical function
    # -------------------------------------------------------------------------
    print("Learning the OR logical function")
    network.reset()
    samples[0] = (0, 0), 0
    samples[1] = (1, 0), 1
    samples[2] = (0, 1), 1
    samples[3] = (1, 1), 1
    learn(network, samples)

    # Example 2 : AND logical function
    # -------------------------------------------------------------------------
    print("Learning the AND logical function")
    network.reset()
    samples[0] = (0, 0), 0
    samples[1] = (1, 0), 0
    samples[2] = (0, 1), 0
    samples[3] = (1, 1), 1
    learn(network, samples)

    # Example 3 : XOR logical function
    # -------------------------------------------------------------------------
    print("Learning the XOR logical function")
    network.reset()
    samples[0] = (0, 0), 0
    samples[1] = (1, 0), 1
    samples[2] = (0, 1), 1
    samples[3] = (1, 1), 0
    learn(network, samples)
