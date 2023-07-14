import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import TensorBoard
import datetime
import time
import matplotlib.pyplot as plt
import os
from kerastuner import RandomSearch


class Keras_NN:
    def __init__(self, filepath, target_column, categorical_features=[]):
        self.filepath = filepath
        self.target_column = target_column
        self.categorical_features = categorical_features
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.history = None
        self.test_loss = None
        self.test_accuracy = None
        self.tensorboard = TensorBoard(
            log_dir="./logs", histogram_freq=0, write_graph=True, write_images=True
        )

    def load_data(self):
        df = pd.read_csv(self.filepath, na_values="?")
        df.dropna(inplace=True)
        self.df = df
        print("\n\n\n DATAFRAME BEFORE PRE-PROCESSING")
        print(self.df.head())

    def one_hot_encode(self):
        self.df = pd.get_dummies(self.df, columns=self.categorical_features)

    def split_data(self):
        features = self.df.drop(self.target_column, axis=1)
        target = self.df[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features, target, test_size=0.2, random_state=0
        )

    def standardise_data(self):
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        print("\n\n\n DATAFRAME AFTER PRE-PROCESSING")
        print(self.df.head())

    def build_model(self, hp):
        model = Sequential()
        model.add(
            Dense(
                units=32,
                activation="relu",
                input_shape=(self.X_train.shape[1],),
            )
        )
        model.add(
            Dense(
                units=16,
                activation="relu",
            )
        )
        model.add(Dense(units=1, activation="linear"))
        model.compile(
            optimizer=optimizers.Adam(hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])),
            loss="mean_squared_error",
            metrics=["accuracy"],
        )
        return model

    def train_model(self):
        tuner = RandomSearch(
            self.build_model,
            objective="val_loss",
            max_trials=20,  # how many model configurations would you like to test?
            executions_per_trial=3,  # how many trials per variation?
            directory="model_dir",
            project_name="MotorPlus",
        )

        tuner.search_space_summary()

        tuner.search(
            self.X_train,
            self.y_train,
            epochs=100,
            validation_data=(self.X_test, self.y_test),
        )

        print(tuner.results_summary())
        print(tuner.get_best_hyperparameters()[0].values)

        self.model = tuner.get_best_models(num_models=1)[0]

        # Re-train the best model to get the history
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=100,
            validation_data=(self.X_test, self.y_test),
            callbacks=[self.tensorboard],
        )

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        self.test_loss = mean_squared_error(self.y_test, y_pred)
        self.test_accuracy = mean_absolute_error(self.y_test, y_pred)

    def plot_history(self):
        plt.plot(self.history.history["loss"])
        plt.plot(self.history.history["val_loss"])
        plt.title("Model MSE")
        plt.ylabel("MSE")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper right")
        plt.show()

    def predict(self, df):
        return self.model.predict(df)

    def save_model(self):
        if not os.path.exists("models_qa"):
            os.makedirs("models_qa")
        time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        self.model.save(f"models_qa/model_{time_str}.h5")

    def load_model(self, model_file):
        self.model = keras.models.load_model(model_file)

    def run(self):
        self.load_data()
        self.one_hot_encode()
        self.split_data()
        self.standardise_data()
        self.train_model()
        self.evaluate_model()
        self.plot_history()
        self.save_model()


if __name__ == "__main__":
    nn = Keras_NN(
        "Datasets/auto-mpg.csv", target_column="mpg", categorical_features=["origin"]
    )
    nn.run()
