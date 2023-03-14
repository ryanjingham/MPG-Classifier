import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import callbacks
from keras.callbacks import TensorBoard
from keras.utils import plot_model
import tensorflow as tf
import datetime
import time
import matplotlib.pyplot as plt
import os
from kerastuner import BayesianOptimization

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
        self.tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    
    def load_data(self):
        df = pd.read_csv(self.filepath, na_values="?")
        df.dropna(inplace=True)
        df.drop(columns=['car name'], inplace=True)
        self.df = df
    
    def one_hot_encode(self):
        self.df = pd.get_dummies(self.df, columns=self.categorical_features)
    
    def split_data(self):
        features = self.df.drop(self.target_column, axis=1)
        target = self.df[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(features, target, test_size=0.2, random_state=0)
    
    def standardise_data(self):
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
    
    def build_model(self, hp):
        """
        In this function, we are defining a Keras Sequential model with the input shape and hidden layers that will be searched by Keras Tuner using hyperparameters.
        The hp object is the hyperparameter object that is passed by Keras Tuner to this function.

        We are defining two types of hyperparameters here:
            Integer hyperparameters: hp.Int. This hyperparameter is used to specify the number of units in a layer. The minimum value is 32, the maximum is 256, and the step size is 32.
            Choice hyperparameters: hp.Choice. This hyperparameter is used to specify the activation function for a layer and learning rate for optimizer.
        
        After defining the model architecture, we compile it with the Adam optimizer and mean squared error loss function."""
        
        model = Sequential()
        model.add(Dense(units=hp.Int('units_input', min_value=32, max_value=256, step=32),
                        activation=hp.Choice('act_input', values=['relu', 'sigmoid']),
                        input_shape=(self.X_train.shape[1],)))
        
        for i in range(hp.Int('num_layers', 1, 5)):
            model.add(Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=256, step=32),
                            activation=hp.Choice('act_' + str(i), values=['relu', 'sigmoid'])))
        
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3])),
                    loss='mean_squared_error', metrics=['accuracy'])
        
        self.model = model

    # def tune_hyperparameters(self):
    #     param_grid = {
    #         'hidden_layer_size': [8, 16, 32],
    #         'learning_rate': [0.01, 0.001, 0.0001],
    #     }
    #     model = KerasRegressor(build_fn=self.build_model, epochs=40, batch_size=32, verbose=0)
    #     self.grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    #     self.grid_search.fit(self.X_train, self.y_train)
    #     print("Best hyperparameters found:", self.grid_search.best_params_)

    def train_model(self):
        self.history = self.model.fit(self.X_train, self.y_train, epochs=40, batch_size=32, callbacks=[self.tensorboard], verbose=0)
    
    def evaluate_model(self):
        self.test_loss = self.model.evaluate(self.X_test, self.y_test, verbose=0)
    
    def predict(self, df):
        return self.model.predict(df)
    
    def plot_history(self):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['accuracy'])
        plt.title('Model Loss and Accuracy')
        plt.ylabel('Loss / Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Loss', 'Accuracy'], loc='upper right')
        plt.show()
    
    def save_model(self):
        if not os.path.exists('models_qa'):
            os.makedirs('models_qa')
        time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
        self.model.save(f"models_qa/model_keras_latest.h5")
    
    def load_model(self, model_file):
        self.model = keras.models.load_model(model_file)
    
    def run(self):
        self.load_data()
        self.one_hot_encode()
        self.split_data()
        self.standardise_data()
        #self.tune_hyperparameters()
        #best_params = self.grid_search.best_params_
        
        tuner = BayesianOptimization(
            self.build_model,
            objective='val_loss',
            max_trials=5,
            executions_per_trial=3,
            directory='keras_tuner_dir',
            project_name='MotorPlusMPG'
        )
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5)
        tuner.search(self.X_train, self.y_train, epochs=40, batch_size=32, validation_data=(self.X_test, self.y_test), callbacks=[early_stop])
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.build_model(best_hps)
        self.train_model()
        self.evaluate_model()
        self.plot_history()
        self.save_model()
        

if __name__ == '__main__':
    nn = Keras_NN('Datasets/auto-mpg.csv', target_column='mpg', categorical_features=['origin'])
    nn.run()