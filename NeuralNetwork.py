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
        # Initialize an array to keep track of the errors for each epoch
        errors = []
        # Loop through each epoch
        for epoch in range(epochs):
            logging.debug(f"Epoch {epoch+1}/{epochs}")
            # Feed forward through the network
            # Initialize the activations with the input layer
            activations = [X]
            # Initialize the list of weighted sums (z)
            zs = []
            # Loop through each layer
            for i in range(len(self.weights)):
                # Calculate the weighted sum for the current layer
                z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
                # Apply the sigmoid activation function to the weighted sum
                a = self.sigmoid(z)
                # Append the activation to the list of activations
                activations.append(a)
                # Append the weighted sum to the list of zs
                zs.append(z)

            # Compute the error
            error = activations[-1] - y
            logging.debug(f"Error: {error}")
            # Append the mean absolute error to the list of errors
            errors.append(np.mean(np.abs(error)))

            # Backpropagation
            # Compute the delta for the output layer
            delta = error * self.sigmoid_derivative(activations[-1])
            # Loop through each layer in reverse order
            for i in range(len(self.weights)-1, -1, -1):
                # Update the weights for the current layer
                self.weights[i] -= self.learning_rate * np.dot(activations[i].T, delta)
                # Update the biases for the current layer
                self.biases[i] -= self.learning_rate * np.sum(delta, axis=0, keepdims=True)
                # Compute the delta for the current layer
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(activations[i])

        # Return the list of errors for each epoch
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