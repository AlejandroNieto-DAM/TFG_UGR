from nodes.DQN.networks import QNetwork
from nodes.DQN.memory import ReplayBuffer
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import rospy

class DQN():
    def __init__(self, fc1_dims = 256, fc2_dims = 256, n_actions = 5, epsilon_min = 0.01, gamma = 0.99, lr = 0.0003, epsilon = 1.0, max_size = 100000, input_dims=[364], batch_size = 64):
        
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.batch_size = batch_size

        self.epsilon = epsilon
        self.epsilon_decay = gamma
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.gamma = gamma

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.model = QNetwork(fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims, n_actions=self.n_actions, name="model")
        self.target_model = QNetwork(fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims, n_actions=self.n_actions, name="target_model")
       
        self.model.compile(optimizer=Adam(learning_rate=self.lr))
        self.target_model.compile(optimizer=Adam(learning_rate=self.lr))

    def store_data(self, states, actions, rewards, new_states, dones):
        self.memory.store_data(states, actions, rewards, new_states, dones)

    def save_models(self):
        print("SAVING MODELS")
        self.model.save_weights(self.model.save_directory)
        self.target_model.save_weights(self.target_model.save_directory)

    def load_models(self):
        self.model.load_weights(self.model.save_directory)
        self.target_model.load_weights(self.target_model.save_directory)

    def choose_action(self, observation):

        q_values = self.model(tf.reshape(observation, (1, observation.shape[0])))

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        return np.argmax(q_values)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def learn(self):
        
        if len(self.memory.states) < self.batch_size:
            return
        
        
        state_arr, action_arr, reward_arr, new_state_arr, dones_arr = self.memory.generate_data(self.batch_size)

        states = tf.convert_to_tensor(state_arr, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state_arr, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward_arr, dtype=tf.float32)
        actions = tf.convert_to_tensor(action_arr, dtype=tf.int32)
        dones = tf.convert_to_tensor(dones_arr, dtype=tf.float32)
        

        with tf.GradientTape() as tape:
            q_values = self.model(states)
            next_q_values = self.target_model(states_)

            max_next_q_values = tf.reduce_max(next_q_values, axis=1)

            y = rewards + self.gamma * max_next_q_values * (1-dones)

            one_hot_actions = tf.one_hot(actions, self.n_actions)
            q_values = tf.reduce_sum(q_values * one_hot_actions, axis=1)

            loss =  tf.keras.losses.MSE(y, q_values)
            
        model_grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(model_grads, self.model.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)