from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
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
        return df

    def create_model(self, layer_size):
        input_size = Input(shape=(self.df.shape[1],))
        train, test = train_test_split(self.df, test_size=0.2)
        # inp_size = self.df.shape[1]

        encoded = Dense(layer_size, activation='relu')(input_size)
        decoded = Dense(self.df.shape[1], activation='sigmoid')(encoded)
        self.autoencoder = Model(input_size, decoded)
        
        self.encoder = Model(input_size, encoded) #map input to small dimension
        encoded_input = Input(shape=(layer_size,)) # 
        
        decoder_layer = self.autoencoder.layers[-1]
        self.decoder = Model(encoded_input, decoder_layer(encoded_input))
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        #train
        self.autoencoder.fit(train, train, epochs=200, batch_size=32, shuffle=True, validation_data=(test, test))
        encoded_imgs = self.encoder.predict(test)
        decoded_imgs = self.decoder.predict(encoded_imgs)
        print(decoded_imgs[:1,:10])


ae = AutoEncoder()