import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization, Activation
import os
from tensorflow.keras import Model
import keras

keras.backend.set_image_data_format('channels_first')

class CNNQNetwork(Model):
    def __init__(self, input_shape, n_actions, name, save_directory = '/model_weights/dqn_cnn/'):
        super(CNNQNetwork, self).__init__()
        self.n_actions = n_actions

        #print("MIRA LA INPUT SHAPE ", input_shape)
        #print("GPUs available: ", len(tf.config.experimental.list_physical_devices('GPU')))

        self.conv1 = Conv2D(32, (8,8), strides = 4, activation='relu', padding='same', input_shape=input_shape, data_format="channels_first")
        self.conv2 = Conv2D(64, (4,4), strides = 2, activation='relu', padding='same')
        self.conv3 = Conv2D(64, (3, 3), strides = 1, activation='relu', padding='same')
        self.conv4 = Conv2D(64, (3, 3), strides = 1, activation='relu', padding='same')
        self.flatten = Flatten()
        self.fc1 = Dense(256, activation='relu')
        self.fc2 = Dense(n_actions, activation='linear') 

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)
