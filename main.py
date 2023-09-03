import sys

from src.network import NeuralNetwork, NetworkState
from src.utils.cache import write_state, read_state


def usage():
    return """
Neural Network - Number Recognizition

Pass a command as first argument to determine the script's behavior.

    init
        Create the cache directory, initiate a random starting state.

    train
        Train the neural network.

    guess
        Use the neural network to analyse a given input.
"""


if len(sys.argv) < 2:
    print(usage())
    exit(1)

match sys.argv[1]:
    case 'init':
        network = NeuralNetwork()

        write_state(network.get_state())

        state = read_state()
        isinstance(state, NetworkState)
        network.set_state(state)

    case 'train':
        pass

    case 'guess':
        pass

    case _:
        print(usage())
        exit(1)
