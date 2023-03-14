import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import TensorBoard
from keras.utils import plot_model
import tensorflow as tf
import datetime
import time
import matplotlib.pyplot as plt
import os

def load_data(filepath):
    df = pd.read_csv(filepath, na_values="?")
    df.dropna(inplace=True)
    df.drop(columns=['car name'], inplace=True)
    return df

def one_hot_encode(df, categorical_features):
    return pd.get_dummies(df, columns=categorical_features)

def split_data(df, target_column):
    features = df.drop(target_column, axis=1)
    target = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

def standardise_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def build_model(input_shape):
    model = Sequential()
    model.add(Dense(8, activation='relu', input_shape=(input_shape.shape[1],)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))
    adam = optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train):
    history = model.fit(X_train, y_train, epochs=40, batch_size=32, callbacks=[tensorboard], verbose=0)
    return history

def evaluate_model(model, X_test, y_test):
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    return test_loss

def predict(df):
    return model.predict(df)

def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['accuracy'])
    plt.title('Model Loss and Accuracy')
    plt.ylabel('Loss / Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Loss', 'Accuracy'], loc='upper right')
    plt.show()

def save_model(model):
    if not os.path.exists('models_qa'):
        os.makedirs('models_qa')
    time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    model.save(f"models_qa/model_keras_latest.h5")

def load_model(model_file):
    return keras.models.load_model(model_file)

if __name__ == "__main__":
    filepath = 'Datasets/auto-mpg.csv'
    target_column = 'mpg'
    categorical_features = ['origin']
    df = load_data(filepath)
    df = one_hot_encode(df, categorical_features)
    X_train, X_test, y_train, y_test = split_data(df, target_column)
    X_train, X_test = standardise_data(X_train, X_test)
    model = build_model(X_train)
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    history = train_model(model, X_train, y_train)
    test_loss = evaluate_model(model, X_test, y_test)
    plot_history(history)
    save_model(model)