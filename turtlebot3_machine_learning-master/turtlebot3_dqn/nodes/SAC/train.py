import rospy
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from nodes.SAC.agent import SAC
import numpy as np

class TrainSAC:
    def __init__(self, state_size = [364],  action_size = 5, episodes = 300, N = 1, lr = 0.03, batch_size = 64,  env = None, using_camera = 0):

        self.using_camera = using_camera
        self.state_size = state_size
        self.action_size = action_size
        self.N = N
        self.n_steps = 0
        self.learn_iters = 0
        self.score_history = []

        self.best_score = 1600
        self.stuck_on_same_avg_score = 0

        self.episodes = episodes

        self.env = env

        self.agent = SAC(input_dims = state_size, lr = lr, batch_size = batch_size, using_camera = self.using_camera)
        self.timestep = 0 

        self.score_history = []
        
        self.figure_file = 'plots/sac_ann_learning_curve.png'

        self.name = "SAC_ANN"
        if self.using_camera:
            self.name = "SAC_CNN"

    def train(self):

        rospy.loginfo(f"TRAINING WITH {self.name} NETS")

        for e in range(self.episodes):
            
            done = False
            state = self.env.reset()
            score = 0
            self.timestep = 0

            while not done:

                action = self.agent.choose_action(state)
                state_, reward, done = self.env.step(action)

                self.agent.store_data(state, action, reward, state_, done)
                
                #rospy.loginfo("Action --> " + str(action) + " Reward --> " + str(reward))

                state = state_
                score += reward
                
                self.timestep += 1
                if self.timestep >= 500:
                    #rospy.loginfo("Time out!!")
                    done = True

                if done:
                    break
            
            self.score_history.append(score)
            avg_score = np.mean(self.score_history[-100:])

            self.env.pause_simulation()
            c1_loss, c2_loss, a_loss, alpha_loss = self.agent.learn()
            self.env.unpause_proxy()
            
            if avg_score == self.best_score:
                self.stuck_on_same_avg_score += 1

            if avg_score > self.best_score or self.stuck_on_same_avg_score == 5:
                self.best_score = avg_score
                #break

            #if e % 10 == 0:
            print('episode', e, 'avg score %.1f' % avg_score, 'learning_steps', self.timestep)
        
        rospy.loginfo("FIN TRAINING " + str(self.best_score))
        return self.score_history