import unittest
import pandas as pd
import numpy as np
import keras
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

class TestKerasModel(unittest.TestCase):
    def setUp(self):
        self.model_file = 'models_qa/model_keras_latest.h5'
        self.df = pd.read_csv('Datasets/auto-mpg.csv', na_values="?")
        self.df.dropna(inplace=True)
        self.df.drop(columns=['car name'], inplace=True)
        self.df = pd.get_dummies(self.df, columns=['origin'])
        self.scaler = StandardScaler()
        self.scaler.fit(self.df.drop('mpg', axis=1))

    def test_model_predictions(self):
        model = load_model(self.model_file)
        X_test = self.scaler.transform(self.df.drop('mpg', axis=1).iloc[:10, :])
        y_test = self.df['mpg'].iloc[:10]
        predictions = model.predict(X_test)
        for i in range(len(predictions)):
            self.assertAlmostEqual(predictions[i][0], y_test.iloc[i], delta=0.1)

if __name__ == '__main__':
    unittest.main()
