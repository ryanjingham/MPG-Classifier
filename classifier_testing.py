import unittest
import pandas as pd
import numpy as np
from NeuralNetwork import NeuralNetwork

class TestClassifier(unittest.TestCase):
    def setUp(self):
        # Load the test data
        data = pd.read_csv("vehicles.csv")

        # Extract the input and output data
        X = data.drop("combination_mpg", axis=1)
        y = data[["combination_mpg"]]

        # Preprocess the data
        X = pd.get_dummies(X)
        X = (X - X.mean()) / X.std()

        self.X_test = X.values
        self.y_test = y.values

    def test_predict(self):
        # Load the trained model
        filename = "models/model_2023-01-22-17-30.pkl"
        nn = NeuralNetwork.load(filename)

        # Use the predict function to make predictions
        y_pred = nn.predict(self.X_test)

        # Check if the predictions are close to the actual values
        np.testing.assert_allclose(y_pred, self.y_test, rtol=1e-05, atol=1e-08)

if __name__ == '__main__':
    unittest.main()