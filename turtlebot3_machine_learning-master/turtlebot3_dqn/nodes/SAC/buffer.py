import numpy as np
import tensorflow as tf
import cv2
import os
import torch
class ReplayBuffer():
    def __init__(self, batch_size, using_camera, shape):
        
        self.using_camera = using_camera
        self.batch_size = batch_size

        
        if not self.using_camera: 
            self.shape = [shape]
        else:
            self.shape = shape
        
        self.mem_size = 10000
        self.counter = 0
    
        self.states = np.zeros((self.mem_size, *self.shape))
        self.new_states =np.zeros((self.mem_size, *self.shape))
        self.actions = np.zeros((self.mem_size))
        self.rewards = np.zeros((self.mem_size))
        self.dones = np.zeros((self.mem_size))


    def get_data(self):

        states = torch.FloatTensor(np.array(self.states)).detach()
        next_states = torch.FloatTensor(np.array(self.new_states)).detach()
        rewards = torch.FloatTensor(np.array(self.rewards)).detach()
        actions = torch.FloatTensor(np.array(self.actions)).detach()
        dones = torch.FloatTensor(np.array(self.dones)).detach()
        

        return states, actions, rewards, next_states, dones
    
    def generate_batches(self):
        max_mem = min(self.counter, self.mem_size)
        batches = []
        for i in range(5):
            batches.append(np.random.choice(max_mem, self.batch_size))

        return batches


    def store_data(self, state, action, reward, new_state, done):
        
        index = self.counter % self.mem_size

        self.states[index] = state
        self.new_states[index] = new_state
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = done

        self.counter += 1

        
    def clear_data(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.new_states = []
        self.dones = []