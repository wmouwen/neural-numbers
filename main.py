import sys
import src.utils.cache as cache
import src.utils.dataset as dataset
from src.network import NeuralNetwork


def usage():
    return """
Neural Network - Number Recognition

Pass a command as first argument to determine the script's behavior.

    load_data
        Download test dataset from the internet.

    init
        Create the cache directory, initiate a random starting state.

    train
        Train the neural network.

    test
        Test the neural network against a different dataset.
"""


if len(sys.argv) < 2:
    print(usage())
    exit(1)

match sys.argv[1]:
    case 'load_data':
        dataset.download()

        print('Data loaded!')

    case 'init':
        network = NeuralNetwork()
        cache.write_state(network.get_state())

        print('Random state generated!')

    case 'train':
        network = NeuralNetwork(state=cache.read_state())

        images = dataset.load('train-images.gz')
        labels = dataset.load('train-labels.gz', label=True)

        for index in range(0, len(images)):
            image = images[index] / 255.0
            correct_answer = labels[index][0]

            network.set_input(image)
            network.run()
            guess = network.get_best_guess()

            if guess == correct_answer:
                print(f"Guess: {guess}, Answer: {correct_answer}")
            else:
                print(f"Guess: {guess}, Answer: {correct_answer} - Incorrect")

        print(f"Done! n={len(images)}")

    case 'test':
        network = NeuralNetwork(state=cache.read_state())

        images = dataset.load('test-images.gz')
        labels = dataset.load('test-labels.gz', label=True)

        correct = 0

        for index in range(0, len(images)):
            image = images[index] / 255.0
            correct_answer = labels[index][0]

            network.set_input(image)
            network.run()
            guess = network.get_best_guess()

            if guess == correct_answer:
                print(f"Guess: {guess}, Answer: {correct_answer}")
                correct += 1
            else:
                print(f"Guess: {guess}, Answer: {correct_answer} - Incorrect")

        print(f"Score: {round(100 * correct / len(images), ndigits=2)}%, n={len(images)}")

    case _:
        print(usage())
        exit(1)
