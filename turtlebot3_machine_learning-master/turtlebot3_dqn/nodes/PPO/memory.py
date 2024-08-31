import numpy as np
import tensorflow as tf
import cv2
import rospy

class Memory:
    def __init__(self, batch_size, using_camera):
        self.using_camera = using_camera
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_data(self):
        indices_muestras = np.arange(len(self.states))
        np.random.shuffle(indices_muestras)
        num_batches = int(len(self.states) / self.batch_size)
        indices_batches = []
        
        for i_batch in range(num_batches):
            indices_batches.append(indices_muestras[self.batch_size * i_batch : self.batch_size * i_batch + self.batch_size])

        if len(self.states) % self.batch_size != 0:     
            indices_batches.append(indices_muestras[self.batch_size * (num_batches) : ])

        return np.array(self.states), np.array(self.probs), np.array(self.vals), np.array(self.actions), np.array(self.rewards), np.array(self.dones), indices_batches

    def store_data(self, state, prob, val, action, reward, done):

        self.states.append(state)
        self.probs.append(prob)
        self.vals.append(val)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear_data(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

    
    


