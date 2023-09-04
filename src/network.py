import numpy as np

Layer = np.ndarray[float]
Weights = np.matrix[float]
Biases = np.ndarray[float]

INPUT_COUNT = 28 * 28
INTERMEDIATE_LAYER_SIZES = [
    16,  # First intermediate layer
    16,  # Second intermediate layer
    10  # Output layer
]
BIAS_SIZE = 10


class NetworkState:
    weights: list[Weights]
    biases: list[Biases]

    def __init__(self, weights: list[Weights], biases: list[Biases]):
        self.weights = weights
        self.biases = biases


class NeuralNetwork:
    layers: list[Layer]
    weights: list[Weights]
    biases: list[Biases]

    def __init__(self, state: NetworkState | None = None):
        # Create initial layer
        self.layers = [np.zeros(INPUT_COUNT)]
        self.weights = []
        self.biases = [np.zeros(len(self.layers[0]))]

        # Create intermediate layers and output layer
        previous_layer_size = len(self.layers[0])
        for layer_size in INTERMEDIATE_LAYER_SIZES:
            # Add layer
            self.layers.append(np.zeros(layer_size))

            # Add weights for incoming edges
            self.weights.append(np.asmatrix(np.random.rand(previous_layer_size, layer_size)))

            # Add biases for layer
            self.biases.append(-1 * BIAS_SIZE * np.random.rand(layer_size))

            previous_layer_size = layer_size

        if isinstance(state, NetworkState):
            self.set_state(state)

    def get_state(self) -> NetworkState:
        return NetworkState(self.weights, self.biases)

    def set_state(self, state: NetworkState) -> None:
        # Check state validity
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

        # State is valid, overwrite weights and biases
        self.weights = state.weights
        self.biases = state.biases

    def set_input(self, inputs: Layer):
        self.layers[0] = inputs
        pass

    def run(self):
        for index in range(1, len(self.layers)):
            signals = self.layers[index - 1]
            weights = self.weights[index - 1]
            biases = self.biases[index]

            absolute_layer = weights.transpose().dot(signals) + biases
            sigmoid_layer = 1 / (1 + np.exp(-1 * absolute_layer))

            self.layers[index] = np.squeeze(np.asarray(sigmoid_layer))

    def cost(self, correct_answer: int):
        tmp = np.copy(self.layers[-1])
        tmp[correct_answer] = 1.0 - tmp[correct_answer]
        return np.sum(tmp * tmp)

    def get_best_guess(self):
        return self.layers[-1].argmax()
