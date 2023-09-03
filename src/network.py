import numpy as np

INTERMEDIATE_LAYER_SIZES = [16, 16]

Weights = np.ndarray[np.ndarray[float]]
Biases = np.ndarray[float]


class NetworkState:
    weights: list[Weights]
    biases: list[Biases]

    def __init__(self, weights: list[Weights], biases: list[Biases]):
        self.weights = weights
        self.biases = biases


class NeuralNetwork:
    inputs: np.ndarray[float]
    layers: list[np.ndarray[float]]
    weights: list[Weights]
    biases: list[Biases]
    outputs: np.ndarray[float]

    def __init__(self):
        self.inputs = np.random.rand(28 * 28)
        self.layers = []
        self.weights = []
        self.biases = []

        previous_layer_size = len(self.inputs)
        for layer_size in INTERMEDIATE_LAYER_SIZES:
            self.layers.append(np.random.rand(layer_size))
            self.weights.append(np.random.rand(previous_layer_size, layer_size))
            self.biases.append(np.random.rand(layer_size))
            previous_layer_size = layer_size

        self.layers.append(np.random.rand(10))
        self.weights.append(np.random.rand(previous_layer_size, 10))
        self.biases.append(np.random.rand(10))
        self.outputs = self.layers[len(self.layers) - 1]

    def set_state(self, state: NetworkState) -> None:
        if len(self.weights) != len(state.weights):
            raise RuntimeError('Invalid state')

        for layer in range(0, len(self.weights)):
            if len(self.weights[layer]) != len(state.weights[layer]):
                raise RuntimeError('Invalid state')

            for source in range(0, len(self.weights)):
                if len(self.weights[layer][source]) != len(state.weights[layer][source]):
                    raise RuntimeError('Invalid state')

        if len(self.biases) != len(state.biases):
            raise RuntimeError('Invalid state')

        for layer in range(0, len(self.biases)):
            if len(self.biases[layer]) != len(state.biases[layer]):
                raise RuntimeError('Invalid state')

        self.weights = state.weights
        self.biases = state.biases

    def get_state(self) -> NetworkState:
        return NetworkState(self.weights, self.biases)
