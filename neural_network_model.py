import json
import os

import numpy as np

MODEL_FILE_NAME = "./model"
MODEL_FILE_EXTENSION = ".json"

NUMBER_OF_LAYERS = 2


class TextClassifierNetworkModel:
    def __init__(self, layer_dimensions, embeddings_matrix, learning_rate=0.01):
        """
        layer_dimensions is a list that must contain the number of neurons in the respective layers: input layer
        (flattened) and hidden layer. For example, "[2300, 128]" means that the network has 2300 neurons in the input
        layer and 128 neurons in the hidden layer. For simplicity, the output layer always contains a single neuron.
        Weights are initialized randomly (for hidden and output layers only) based on the number of neurons (Glorot).
        Biases are all zeros by default. All the layers in this network are dense. The activation function is Sigmoid.
        The loss function is the Mean Squared Error (MSE). The optimizer is the Stochastic Gradient Descent (SGD).
        embeddings_matrix is used to convert tokenized word sequences into vector embeddings (flattened for the input
        layer).

        The goal of this code is visual simplicity, thus it's not optimized and really slow. Overall, it's a great
        exercise, but also a great example of why it's better to use Tensorflow.
        """

        assert len(layer_dimensions) == NUMBER_OF_LAYERS
        self.layer_dimensions = layer_dimensions
        self.embeddings_matrix = embeddings_matrix
        self.learning_rate = learning_rate
        self.model_file_path = MODEL_FILE_NAME + str(layer_dimensions) + MODEL_FILE_EXTENSION

        if os.access(self.model_file_path, os.R_OK):
            print("Loading model from a file...")
            with open(self.model_file_path, 'r') as file:
                data = json.load(file)
                self.weights = data[0]
                self.biases = data[1]

                # Because json doesn't deserialize anything into numpy data types
                for layer in range(NUMBER_OF_LAYERS):
                    self.weights[layer] = np.array(self.weights[layer])
                    self.biases[layer] = np.array(self.biases[layer])

        else:
            print("No model file found! Initializing default weights...")
            low_bound_h, high_bound_h = glorot_weight_bounds(layer_dimensions[0], layer_dimensions[1])
            low_bound_o, high_bound_o = glorot_weight_bounds(layer_dimensions[1], 1)

            # [(128, 2300), (128,)] for the given example
            self.weights = [np.random.uniform(low=low_bound_h,
                                              high=high_bound_h,
                                              size=(layer_dimensions[1], layer_dimensions[0])),
                            np.random.uniform(low=low_bound_o,
                                              high=high_bound_o,
                                              size=layer_dimensions[1])]

            # [(128,), (1,)] for the given example
            self.biases = [np.zeros(layer_dimensions[1]),
                           np.zeros(1)]

        print(self)

    def __repr__(self):
        return "Network Structure: {}".format(self.layer_dimensions)

    def predict(self, validation_sequences):
        validation_sequences = convert_to_embeddings_and_flatten(validation_sequences, self.embeddings_matrix)
        classifications = []
        for k in range(len(validation_sequences)):
            ah = sigmoid(np.dot(self.weights[0], validation_sequences[k]) + self.biases[0])
            ao = sigmoid(np.dot(self.weights[1], ah) + self.biases[1])
            classifications.append(ao)
        return classifications

    def fit(self, training_sequences, labels):
        training_sequences = convert_to_embeddings_and_flatten(training_sequences, self.embeddings_matrix)
        sample_size = len(training_sequences)

        der_e_w = [np.zeros((self.layer_dimensions[1], self.layer_dimensions[0])),
                   np.zeros(self.layer_dimensions[1])]
        der_e_b = [np.zeros(self.layer_dimensions[1]), np.zeros(1)]
        e = 0.0

        for k in range(sample_size):

            # Forward pass
            zh = np.dot(self.weights[0], training_sequences[k]) + self.biases[0]
            ah = sigmoid(zh)
            zo = np.dot(self.weights[1], ah) + self.biases[1]
            ao = sigmoid(zo)

            # Calculate the error
            ek = np.power(ao - labels[k], 2)
            e += ek

            # Partial derivatives and backward pass
            der_ek_ao = 2.0 * (ao - labels[k])
            der_ao_zo = sigmoid_prime(zo)
            der_zo_ah = self.weights[1]
            der_ah_zh = sigmoid_prime(zh)

            der_zh_wh = training_sequences[k]
            der_zo_wo = ah

            # For the hidden layer
            # (128, 2300) += (2300,) * (128, 1) * (128, 1) * (1) * (1) for the given example
            der_e_w[0] += der_zh_wh * der_ah_zh[:, None] * der_zo_ah[:, None] * der_ao_zo * der_ek_ao
            der_e_b[0] += der_ah_zh * der_zo_ah * der_ao_zo * der_ek_ao

            # For the output layer
            der_e_w[1] += der_zo_wo * der_ao_zo * der_ek_ao
            der_e_b[1] += der_ao_zo * der_ek_ao

        # Averaging errors and adjusting the learning rate
        print("Loss: %f" % (e / sample_size))
        for layer in range(NUMBER_OF_LAYERS):
            der_e_w[layer] /= sample_size
            der_e_b[layer] /= sample_size

        # Updating weights and biases
        for layer in range(NUMBER_OF_LAYERS):
            self.weights[layer] -= self.learning_rate * der_e_w[layer]
            self.biases[layer] -= self.learning_rate * der_e_b[layer]

        # Serializing everything to a file, so I can cancel it anytime and start from that moment
        with open(self.model_file_path, 'w') as outfile:
            json.dump([self.weights, self.biases], outfile, cls=NumpyEncoder)


def glorot_weight_bounds(prev_layer_size, next_layer_size):
    return (-(np.sqrt(6.0) / np.sqrt(prev_layer_size + next_layer_size)),
            (np.sqrt(6.0) / np.sqrt(prev_layer_size + next_layer_size)))


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def convert_to_embeddings_and_flatten(sequences, embedding_matrix):
    flattened_sequences = []
    for sequence in sequences:
        flattened = []
        for word in sequence:
            vector = embedding_matrix[word]
            for value in vector:
                flattened.append(value)
        flattened_sequences.append(np.array(flattened))
    return np.array(flattened_sequences)


# JSON Numpy Encoder for serializing weights and biases
# https://stackoverflow.com/questions/3488934/simplejson-and-numpy-array/24375113#24375113
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
