import numpy as np
import cv2
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import rospy
import matplotlib.pyplot as plt

class ReplayBuffer():
    def __init__(self, max_size, shape, n_actions):
        # This * 5 its cause we will generate 5 new images per image
        # using data augmentation so we have to increase our memory 
        # in order to learn with all that images
        self.mem_size = 20000
        self.counter = 0
        self.n_actions = n_actions

        print(shape, type(shape))
 
        self.states = np.zeros((self.mem_size, *shape))
        self.new_states = np.zeros((self.mem_size, *shape))

        self.actions = np.zeros((self.mem_size))

        self.rewards = np.zeros((self.mem_size))
        self.dones = np.zeros((self.mem_size))


    def load_images_batch(self, states, batch_size):

        imgs = np.zeros((batch_size, *(84,84,3)))

        for i, state in enumerate(states):
            img = self.load_and_convert_image(state)
            #rospy.loginfo("MIRAMIRA -- " + str(img))
            imgs[i] = img

        return imgs

        
    def generate_data(self, batch_size):
        max_mem = min(self.counter, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)
        
        #states = self.load_images_batch(self.states[batch], batch_size)
        #new_states = self.load_images_batch(self.new_states[batch], batch_size)
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
    
    def clear_data(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.new_states = []
        self.dones = []

    def load_and_convert_image(self, image):
        image_path = f"/home/nietoff/tfg/src/turtlebot3_machine_learning-master/turtlebot3_dqn/images/ppo_images/{image}"
        
        image = cv2.imread(image_path) 
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow('Im', rgb_image)
        cv2.waitKey(0)
        #rgb_image = tf.image.convert_image_dtype(rgb_image, tf.float32)
        #resized_image = tf.image.resize(rgb_image, [84, 84], method=tf.image.ResizeMethod.BILINEAR)
        
        return rgb_image

    def augment_image(self, image, num_augments=5):
        #image = tf.expand_dims(image, axis=0)
        augmented_images = []
        for _ in range(num_augments):
            augmented_image = self.datagen.flow(image, batch_size=1)[0]
            augmented_images.append(tf.squeeze(augmented_image, axis=0))
        return augmented_images
    
          