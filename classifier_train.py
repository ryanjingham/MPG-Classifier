import itertools
import numpy as np
import pandas as pd
from NeuralNetwork import NeuralNetwork
import time

# Load the training data
data = pd.read_csv("vehicles.csv")

# Extract the input and output data
X = data.drop("combination_mpg", axis=1)
print(X)
y = data[["combination_mpg"]]

# Preprocess the data
# One-hot encode the categorical variables
X = pd.get_dummies(X)
# Normalize the numeric variables
X = (X - X.min()) / (X.max() - X.min()) * 2 - 1


# Define the hyperparameters to search over
hidden_layer_sizes = [[10, 10], [20, 20], [30, 30]]
learning_rates = [0.1, 0.01, 0.001]
epochs = [100, 200, 300]

# Create a list of all combinations of hyperparameters
combinations = list(itertools.product(hidden_layer_sizes, learning_rates, epochs))

# Initialize a list to store the models and their performance
models = []

# Train a model with each combination of hyperparameters
for hidden_layer_size, learning_rate, epoch in combinations:
    # Train the model
    input_shape = X.shape[1]
    output_shape = 1 
    nn = NeuralNetwork(input_shape, hidden_layer_size, output_shape, learning_rate)
    nn.train(X.values, y.values, epoch)

    # Evaluate the model's performance
    predictions = nn.predict(X.values)
    error = np.mean(np.abs(predictions - y.values))

    # Store the model and its performance
    models.append((nn, error))

# Select the model with the best performance
best_model = min(models, key=lambda x: x[1])

# Save the model
current_time = time.strftime("%Y-%m-%d-%H-%M", time.gmtime())
filename = "models/model_" + current_time + ".pkl"
nn.save(filename)