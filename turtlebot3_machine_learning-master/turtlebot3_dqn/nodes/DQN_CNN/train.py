import rospy
import os
import json
import numpy as np
import random
import time
import sys
import threading

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32MultiArray
from src.turtlebot3_dqn.environment import Env
from nodes.DQN_CNN.agent import DQN
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Activation
from nodes.utils import plot_learning_curve
import collections
import keras
import pandas as pd

keras.backend.set_image_data_format('channels_first')

state_stack = collections.deque(maxlen=4)


class TrainDQNCNN:
    def __init__(self, state_size = (4,60,80), action_size = 5, episodes = 600, N = 20, lr = 0.003, batch_size = 64, env = None):

        self.state_size = state_size
        self.action_size = action_size
        self.N = N
        self.n_steps = 0
        self.learn_iters = 0
        self.score_history = []

        self.best_score = 200
        self.target_update = 2000

        self.episodes = episodes

        self.env = env

        self.agent = DQN(input_dims=state_size, n_actions = action_size, lr = lr, batch_size = batch_size)
        self.timestep = 0

        self.avg_scores_history = []

        self.figure_file = 'plots/dqn_cnn_learning_curve.png'
        self.first_train = False
        
    def train(self):

        #self.agent.load_models()
        rospy.loginfo("TRAINING DQN WITH CNN NETS")


        for e in range(self.episodes):
            done = False
            state = self.env.reset()
            score = 0
            self.timestep = 0

            for i in range(4):
                state_stack.append(state)

            state = np.asarray(state_stack)


            #rospy.loginfo("Running episode " + str(e))

            
            while not done:

                #if len(state.shape) == 4 :
                # Tensor has an extra dimension (84, 84, 1, 1); remove the extra dimension
                #    state = tf.squeeze(state, axis=0)
                action = self.agent.choose_action(state)
                state_, reward, done = self.env.step(action)
                
                state_stack.append(state_)
                state_ = np.asarray(state_stack)

                self.agent.store_data(state, action, reward, state_, done)

                self.n_steps += 1
                if self.n_steps % self.N == 0:
                    self.env.pause_simulation()
                    loss = self.agent.learn()
                    #rospy.loginfo("LOSS --> " + str(loss))
                    self.env.unpause_proxy()
                    self.first_train = False

                state = state_
                score += reward

                rospy.loginfo("Action --> " + str(action) + " Reward --> " + str(reward))

                self.timestep += 1
                if self.timestep >= 500:
                    #rospy.loginfo("Time out!!")
                    done = True

                if done:
                    break
                
            self.score_history.append(score)
            avg_score = np.mean(self.score_history[-100:])
            
            self.avg_scores_history.append(avg_score)

            if avg_score > self.best_score:
                self.best_score = avg_score
                #self.agent.save_models()
  

            if (e+1) % 5 == 0 and self.first_train:
                self.agent.update_target_model()

            if (e+1) % 3 == 0:
                df = pd.DataFrame()
                dfs_arr = []
                score_history = self.score_history
                dfs_arr.append(pd.DataFrame([score_history], columns=[f'ep{i+1}' for i in range(len(score_history))]))
                df = pd.concat([dfs_arr[0]], ignore_index=True)
                labels = ["BATCH_SIZE=128 LR=0.0003"]
                plot_learning_curve(np.arange(1, 601), df, "/home/nietoff/tfg/src/turtlebot3_machine_learning-master/turtlebot3_dqn/images/", "DQN CNN STAGE 1 ", labels)

            print('episode', e, 'avg score %.1f' % avg_score, 'learning_steps', self.timestep)
        
        return self.score_history
