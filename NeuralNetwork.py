import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


class NeuralNetwork:
    def __init__(
        self, input_shape, hidden_layer_size, output_shape, learning_rate=0.05
    ):
        self.input_shape = input_shape
        self.hidden_layer_size = hidden_layer_size
        self.output_shape = output_shape
        self.learning_rate = learning_rate

        # Initialise the weights and biases for each layer with random values drawn from a normal distribution
        self.weights = [
            np.random.normal(0, 1, size=(self.input_shape, self.hidden_layer_size)),
            np.random.normal(0, 1, size=(self.hidden_layer_size, self.output_shape)),
        ]
        self.biases = [
            np.zeros((1, self.hidden_layer_size)),
            np.zeros((1, self.output_shape)),
        ]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, X, y, epochs):
        errors = []
        for epoch in range(epochs):
            activations = [X]
            zs = []
            for i in range(len(self.weights)):
                # Forward pass
                z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
                a = self.sigmoid(z)
                activations.append(a)
                zs.append(z)

            # Backpropagation
            error = activations[-1] - y
            errors.append(np.mean(np.abs(error)))

            delta = error * self.sigmoid_derivative(activations[-1])
            for i in range(len(self.weights) - 1, -1, -1):
                # Update weights and biases using gradient descent
                self.weights[i] -= self.learning_rate * np.dot(activations[i].T, delta)
                self.biases[i] -= self.learning_rate * np.sum(
                    delta, axis=0, keepdims=True
                )
                if i != 0:
                    delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(
                        activations[i]
                    )

        return errors

    def visualise_errors(self, errors, mae, mse):
        epochs = range(1, len(errors) + 1)
        plt.figure(figsize=(10, 5))

        # Plot Mean Absolute Error (MAE)
        plt.subplot(1, 2, 1)
        plt.plot(epochs, errors, "b.-")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.title(f"Mean Absolute Error (MAE): {mae:.4f}")

        # Plot Mean Squared Error (MSE)
        plt.subplot(1, 2, 2)
        plt.plot(epochs, [e**2 for e in errors], "r.-")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title(f"Mean Squared Error (MSE): {mse:.4f}")

        plt.tight_layout()
        plt.show()

    def predict(self, X):
        activations = [X]
        for i in range(len(self.weights)):
            # Forward pass to make predictions
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
            activations.append(a)
        return activations[-1]
