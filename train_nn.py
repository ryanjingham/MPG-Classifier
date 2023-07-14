import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
from NeuralNetwork import NeuralNetwork

if __name__ == "__main__":
    # Load the dataset and data preprocessing
    data = pd.read_csv("Datasets/auto-mpg.csv")
    scaler = MinMaxScaler()
    data = data.replace("?", np.nan)
    data = data.dropna()

    # Prepare the input features (X) and target variable (y)
    X = data.drop("mpg", axis=1).values
    y = data["mpg"].values.reshape(-1, 1)

    # Scale the input features and target variable
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # training parameters
    input_shape = X_train.shape[1]
    output_shape = 1
    epochs = 500

    # Variables to track the best MAE and corresponding parameters
    best_mae = np.inf
    best_params = {}
    errors_list = []
    mae_list = []
    mse_list = []

    # Loop through different combinations of learning rates and hidden layer sizes
    for learning_rate in [0.01, 0.05, 0.1]:
        for hidden_layer_size in [5, 10, 15]:
            # Create a neural network instance
            neural_network = NeuralNetwork(
                input_shape, hidden_layer_size, output_shape, learning_rate
            )

            # Train the neural network and get the errors
            errors = neural_network.train(X_train, y_train, epochs)
            errors_list.append(errors)

            # Make predictions on the test set
            y_pred = neural_network.predict(X_test)

            # Calculate the Mean Absolute Error (MAE) and Mean Squared Error (MSE)
            mae = mean_absolute_error(
                scaler.inverse_transform(y_test), scaler.inverse_transform(y_pred)
            )
            mse = mean_squared_error(
                scaler.inverse_transform(y_test), scaler.inverse_transform(y_pred)
            )

            mae_list.append(mae)
            mse_list.append(mse)

            # Print the metrics and parameters for each combination
            print(
                f"MAE: {mae} | MSE: {mse} | Params: learning_rate={learning_rate}, hidden_layer_size={hidden_layer_size}"
            )

            # Check if the current combination yields the best MAE
            if mae < best_mae:
                best_mae = mae
                best_params = {
                    "learning_rate": learning_rate,
                    "hidden_layer_size": hidden_layer_size,
                }

    # Plot MAE and MSE for all combinations
    for i in range(len(errors_list)):
        neural_network.visualise_errors(errors_list[i], mae_list[i], mse_list[i])

    # Save the best model
    best_model = neural_network
    model_path = "models_qa/neural_network_model.pkl"
    with open(model_path, "wb") as file:
        pickle.dump(best_model, file)

    # Print the best MAE and corresponding parameters
    print(f"Best MAE: {best_mae} | Best Params: {best_params}")
