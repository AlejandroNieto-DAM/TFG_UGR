import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.distributions import Categorical
import torch.nn.init as init
import rospy
import keras

keras.backend.set_image_data_format('channels_first')

class Actor(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims, fc2_dims, name, save_directory='model_weights/sac/'):
        super(Actor, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.model_name = name
        self.save_directory = os.path.join(save_directory, self.model_name + '_sac')


        self.net = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Linear(fc2_dims, n_actions)
        )

    def forward(self, state):
        x = F.softmax(self.net(state), dim=-1)
        return x
    
    def evaluate(self, state, epsilon=1e-8):
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample()
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * epsilon
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities        
    
    
    def get_det_action(self, state):
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        return action.detach().cpu()
    

class Critic(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims, fc2_dims, name, save_directory='model_weights/sac/'):
        super(Critic, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.model_name = name
        self.save_directory = os.path.join(save_directory, self.model_name + '_sac')


        self.net = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Linear(fc2_dims, n_actions)
        )

    def forward(self, state):        
        return self.net(state)


class CNNCritic(nn.Module):
    def __init__(self, n_actions, name, save_directory='/model_weights/dqn/'):
        super(CNNCritic, self).__init__()
        self.name = name
        self.save_directory = save_directory
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=0),
            nn.PReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
            nn.Flatten(),
            nn.Linear(2112, 256),
            nn.PReLU(),
            nn.Linear(256, n_actions),
        )
        
    def forward(self, x):
        return self.net(x)


class CNNActor(nn.Module):
    def __init__(self, n_actions, name, save_directory='/model_weights/dqn/'):
        super(CNNActor, self).__init__()
        self.name = name
        self.save_directory = save_directory
        self.n_actions = n_actions
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=0),
            nn.PReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
            nn.Flatten(),
            nn.Linear(2112, 256),
            nn.PReLU(),
            nn.Linear(256, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.net(x)
        return x
    
    def evaluate(self, state, epsilon=1e-6):
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample()
        z = action_probs == 0.0
        z = z.float() * epsilon
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities        
    
    def get_det_action(self, state):
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.detach().cpu()