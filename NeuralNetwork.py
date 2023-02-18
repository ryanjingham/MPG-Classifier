import logging
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

logging.basicConfig(filename='classifier_ai.log', level=logging.DEBUG, format="%(message)s")


class NeuralNetwork:
    def __init__(self, input_shape, hidden_layer_sizes, output_shape, learning_rate=0.05):
        self.input_shape = input_shape
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_shape = output_shape
        self.learning_rate = learning_rate

        # Initialize the weights and biases for each layer
        self.weights = []
        self.biases = []
        for i, layer_size in enumerate(self.hidden_layer_sizes + [self.output_shape]):
            if i == 0:
                input_size = self.input_shape
            else:
                input_size = self.hidden_layer_sizes[i-1]
            self.weights.append(np.random.normal(0, 1, (input_size, layer_size)))
            self.biases.append(np.zeros((1, layer_size)))

    def sigmoid(self, x):
        # Sigmoid activation function
        # print(x)
        exp_value = -x + 0.1
        # print(exp_value)
        sig = 1 / (1 + np.exp(exp_value))
        return sig

    def sigmoid_derivative(self, x):
        # Derivative of the sigmoid function
        return x * (1 - x)

    def train(self, X, y, epochs):
        print(f"X : {X[0]}")
        errors = []
        for epoch in range(epochs):
            logging.debug(f"Epoch {epoch+1}/{epochs}")
            # Feed forward through the network
            activations = [X]
            for i in range(len(self.weights)):
                z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
                a = self.sigmoid(z)
                activations.append(a)

            # Compute the error
            error = activations[-1] - y
            logging.debug(f"Error: {error}")
            errors.append(np.mean(np.abs(error)))
            
        return errors
        
        
    def predict(self, X):
        activations = [X]
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
            activations.append(a)
        return activations[-1]    
    
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            nn = pickle.load(f)
        return nn
        
        
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def run_neural_network_training(X_train, y_train, hidden_layer_sizes, epochs):
        # Create a NeuralNetwork instance
        input_shape = X_train.shape[1]
        output_shape = 1
        nn = NeuralNetwork(input_shape, hidden_layer_sizes, output_shape)

        # Train the network
        errors = nn.train(X_train, y_train, epochs)

        return nn