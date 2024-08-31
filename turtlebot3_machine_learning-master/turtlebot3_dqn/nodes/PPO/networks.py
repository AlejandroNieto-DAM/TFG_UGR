import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization, GlobalAveragePooling2D
import os
from tensorflow.keras import Model
import rospy
import numpy as np

class Actor(Model):
    def __init__(self, n_actions, fc1_dims, fc2_dims, name, save_directory = 'model_weights/ppo/'):
        super(Actor, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.model_name = name
        self.save_directory =os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + save_directory, self.model_name + '_ppo')

        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.fc3 = Dense(n_actions, activation='softmax')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class Critic(Model):
    def __init__(self, fc1_dims, fc2_dims, name, save_directory = 'model_weights/ppo/'):
        super(Critic, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.model_name = name
        self.save_directory = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + save_directory, self.model_name + '_ppo')
        
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc1_dims, activation='relu')
        self.fc3 = Dense(1, None)

    def call(self, state):
        output = self.fc1(state)
        output = self.fc2(output)
        output = self.fc3(output)

        return output
