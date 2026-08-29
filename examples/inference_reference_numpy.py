import numpy as np


def single_sample():
    weights = np.array([[1.0, 2.0], [-1.0, 0.5]], dtype=np.float32)
    biases = np.array([[0.5], [-2.0]], dtype=np.float32)
    inputs = np.array([[2.0], [-1.0]], dtype=np.float32)
    output = weights @ inputs + biases
    print("single_sample_output")
    print(output)


def batch():
    weights = np.array([[1.0, 2.0], [-1.0, 0.5]], dtype=np.float32)
    biases = np.array([[0.5], [-2.0]], dtype=np.float32)
    inputs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    output = weights @ inputs + biases
    print("batch_output")
    print(output)


def two_layers():
    weights_1 = np.array([[1.0, -1.0], [0.5, 2.0]], dtype=np.float32)
    biases_1 = np.array([[0.0], [1.0]], dtype=np.float32)
    weights_2 = np.array([[2.0, -3.0]], dtype=np.float32)
    biases_2 = np.array([[0.25]], dtype=np.float32)
    inputs = np.array([[2.0, -1.0], [1.0, 3.0]], dtype=np.float32)

    hidden = np.maximum(weights_1 @ inputs + biases_1, 0.0)
    output = weights_2 @ hidden + biases_2
    print("two_layers_output")
    print(output)


single_sample()
batch()
two_layers()
