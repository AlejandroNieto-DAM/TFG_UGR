import numpy as np
import cv2
import tensorflow as tf
import rospy
import matplotlib.pyplot as plt

class ReplayBuffer():
    def __init__(self, max_size, shape, n_actions):

        self.counter = 0
        self.n_actions = n_actions
        self.mem_size = max_size
        
        self.states = np.zeros((self.mem_size, *shape))
        self.new_states = np.zeros((self.mem_size, *shape))
        self.actions = np.zeros((self.mem_size))
        self.rewards = np.zeros((self.mem_size))
        self.dones = np.zeros((self.mem_size))

        
    def generate_data(self, batch_size):
        max_mem = min(self.counter, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.states[batch]
        new_states = self.new_states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        dones = self.dones[batch]

        return states, actions, rewards, new_states, dones


    def store_data(self, state, action, reward, new_state, done):
        
        index = self.counter % self.mem_size

        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.new_states[index] = new_state
        self.dones[index] = done

        self.counter += 1
    
