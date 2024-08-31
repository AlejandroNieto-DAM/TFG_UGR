import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization, GlobalAveragePooling2D
import os
from tensorflow.keras import Model
import rospy
import numpy as np



class CNNCritic(Model):
    def __init__(self, input_shape, name, save_directory='/model_weights/ppo/'):
        super(CNNCritic, self).__init__(name=name)
        self.conv1 = Conv2D(32, (8,8), strides = 4, activation='relu', padding='same', input_shape=input_shape)
        self.conv2 = Conv2D(64, (4,4), strides = 2, activation='relu', padding='same')
        self.conv3 = Conv2D(64, (3, 3), strides = 1, activation='relu', padding='same')
        self.conv4 = Conv2D(32, (1, 1), strides = 1, activation='relu', padding='same')
        self.flatten = Flatten()
        self.fc1 = Dense(256, activation='relu')
        self.fc2 = Dense(1, None)  # Linear activation for critic output

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)

class CNNActor(Model):
    def __init__(self, input_shape, n_actions, name, save_directory='/model_weights/ppo/'):
        super(CNNActor, self).__init__(name=name)
        self.conv1 = Conv2D(32, (8,8), strides = 4, activation='relu', padding='same', input_shape=input_shape)
        self.conv2 = Conv2D(64, (4,4), strides = 2, activation='relu', padding='same')
        self.conv3 = Conv2D(64, (3, 3), strides = 1, activation='relu', padding='same')
        self.conv4 = Conv2D(32, (1, 1), strides = 1, activation='relu', padding='same')
        self.flatten = Flatten()
        self.fc1 = Dense(256, activation='relu')
        self.fc2 = Dense(n_actions, activation='softmax')  # Linear activation for critic output

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)