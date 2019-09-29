import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import warnings

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class AutoEncoder:
    def __init__(self):
        self.df = self.import_data()
        self.create_model(100) # specify hidden layer size

    def import_data(self):
        path = os.getcwd()+'/x_modified.csv'
        df = pd.read_csv(path)
        print(df.shape)
        return df

    def create_model(self, layer_size):
        train, test = train_test_split(self.df, test_size=0.2)
        input_size = self.df.shape[1]

        inputs = Input(shape=(input_size,))
        encoded = Dense(layer_size)(inputs)
        self.encoder = Model(inputs, encoded) # map input to small dimension
        self.encoder.summary()

        encoded_input = Input(shape=(layer_size,))
        decoded = Dense(input_size)(encoded_input)
        self.decoder = Model(encoded_input, decoded)
        self.decoder.summary()

        self.autoencoder = Model(inputs, self.decoder(self.encoder(inputs)))
        self.autoencoder.summary()
        self.autoencoder.compile(optimizer='adam', loss='mse')

        # train
        self.autoencoder.fit(train, train, epochs=500, batch_size=32, shuffle=True, validation_data=(test, test))
        encoded_imgs = self.encoder.predict(test)
        decoded_imgs = self.decoder.predict(encoded_imgs)
        print(decoded_imgs[:1,:10])


if __name__ == "__main__":
    ae = AutoEncoder()