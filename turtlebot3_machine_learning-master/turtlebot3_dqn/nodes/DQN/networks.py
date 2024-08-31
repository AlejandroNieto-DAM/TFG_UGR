import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
import os
from tensorflow.keras import Model

    
class QNetwork(Model):
    def __init__(self, fc1_dims, fc2_dims, n_actions, name, save_directory = '/model_weights/dqn_ann_stage_1/'):
        super(QNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.model_name = name
        self.save_directory = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + save_directory)
        
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc1_dims, activation='relu')
        self.fc3 = Dense(self.n_actions, activation='linear')

    def call(self, state):
        output = self.fc1(state)
        output = self.fc2(output)
        output = self.fc3(output)

        return output
