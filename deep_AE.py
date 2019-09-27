import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras import optimizers
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pandas as pd
import os

path = os.getcwd()+'/x_modified.csv'
df = pd.read_csv(path)
print(df.head)

train_x, val_x = train_test_split(df, test_size=0.2)
input_shape = (df.shape)
input_layer = df.shape[1]
print(df.shape)

autoencoder = Sequential()
autoencoder.add(Dense(128,  activation='elu', input_shape=(input_layer,)))
autoencoder.add(Dense(64,  activation='elu'))
autoencoder.add(Dense(32,    activation='linear', name="bottleneck"))
autoencoder.add(Dense(64,  activation='elu'))
autoencoder.add(Dense(128,  activation='elu'))
autoencoder.add(Dense(input_layer,  activation='sigmoid'))
autoencoder.compile(loss='mean_squared_error', optimizer = Adam())
trained_model = autoencoder.fit(train_x, train_x, batch_size=256, epochs=50, verbose=1, validation_data=(val_x, val_x))
encoder = Model(autoencoder.input, autoencoder.get_layer('bottleneck').output)
encoded_data = encoder.predict(train_x)  # bottleneck representation

decoded_output = autoencoder.predict(train_x)        # reconstruction
encoding_dim = 4

encoded_df = pd.DataFrame.from_records(encoded_data)
recon_df = pd.DataFrame.from_records(decoded_output)

encoded_df.to_csv(r'x_encoded.csv')
recon_df.to_csv(r'x_reconstructed.csv')

# return the decoder
encoded_input = Input(shape=(encoding_dim,))
decoder = autoencoder.layers[-3](encoded_input)
decoder = autoencoder.layers[-2](decoder)
decoder = autoencoder.layers[-1](decoder)
decoder = Model(encoded_input, decoder)
