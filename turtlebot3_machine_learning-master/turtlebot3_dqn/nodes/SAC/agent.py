from nodes.SAC.networks import Actor, Critic, CNNActor, CNNCritic
from nodes.SAC.buffer import ReplayBuffer
import torch
import torch.distributions as distributions
import numpy as np
import torch.optim as optim
import copy
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import rospy

def init_weights(m):
    if type(m) == torch.nn.Linear:
        fan_in = m.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        return (-lim, lim)


class SAC():
    def __init__(self, fc1_dims = 256, fc2_dims = 256, n_actions = 5, lr = 0.0005, gamma = 0.99, tau = 0.005, input_dims=[364], batch_size = 128, using_camera = 0):

        self.using_camera = using_camera
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.n_epochs = 5
        self.gamma = gamma
        self.tau = tau
        self.target_update_interval = 1
        self.learn_step_counter = 0
        self.clip_grad_param = 1

        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.target_entropy = -n_actions
        
        self.memory = ReplayBuffer(batch_size, using_camera, input_dims)

        if self.using_camera:
            self.policy = CNNActor(conv1_dims=(32, (3, 3)), conv2_dims=(64, (3, 3)), fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims, n_actions = self.n_actions, name = "actor")  
            self.q1 = CNNCritic(input_dims = input_dims, n_actions = 5, conv1_dims=(32, (3, 3)), conv2_dims=(64, (3, 3)), fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims, name = "q1")
            self.q2 = CNNCritic(input_dims = input_dims, n_actions = 5, conv1_dims=(32, (3, 3)), conv2_dims=(64, (3, 3)), fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims, name = "q2")
            self.target_q1 = CNNCritic(input_dims = input_dims, n_actions = 5, conv1_dims=(32, (3, 3)), conv2_dims=(64, (3, 3)), fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims, name = "t_q1")
            self.target_q2 = CNNCritic(input_dims = input_dims, n_actions = 5, conv1_dims=(32, (3, 3)), conv2_dims=(64, (3, 3)), fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims, name = "t_q2")
        else:
            self.policy = Actor(input_dims=input_dims, fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims, n_actions = self.n_actions, name = "actor")
            self.q1 = Critic(input_dims=input_dims, n_actions=self.n_actions, fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims, name = "q1")
            self.q2 = Critic(input_dims=input_dims, n_actions=self.n_actions, fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims, name = "q2")
            self.target_q1 = Critic(input_dims=input_dims, n_actions=self.n_actions, fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims, name = "t_q1")
            self.target_q2 = Critic(input_dims=input_dims, n_actions=self.n_actions, fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims, name = "t_q2")
      
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer=optim.Adam(self.q2.parameters(), lr=lr)
        self.target_q1_optimizer=optim.Adam(self.target_q1.parameters(), lr=lr)
        self.target_q2_optimizer=optim.Adam(self.target_q2.parameters(), lr=lr)
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        
        self.policy.apply(init_weights)
        self.q1.apply(init_weights)
        self.q2.apply(init_weights)

        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        

    def store_data(self, state, action, reward, new_state, done):
        self.memory.store_data(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if self.using_camera:
            observation = torch.from_numpy(np.copy(np.asarray(observation, dtype = np.float32)))

        action = self.policy.get_det_action(torch.FloatTensor(observation).unsqueeze(0))
        return action.numpy()[0]

    def save_models(self):
        self.policy.save_weights(self.policy.save_directory + ".h5")
        self.q1.save_weights(self.q1.save_directory + ".h5")
        self.q2.save_weights(self.q2.save_directory + ".h5")
        self.target_q1.save_weights(self.target_q1.save_directory + ".h5")
        self.target_q2.save_weights(self.target_q2.save_directory + ".h5")

    def load_models(self):
        self.policy.load_weights(self.policy.save_directory + ".h5")
        self.q1.load_weights(self.q1.save_directory + ".h5")
        self.q2.load_weights(self.q2.save_directory + ".h5")
        self.target_q1.load_weights(self.target_q1.save_directory + ".h5")
        self.target_q2.load_weights(self.target_q2.save_directory + ".h5")

    def update_policy(self, states, alpha):
        _ , action_probs, log_pis = self.policy.evaluate(states)

        q1 = self.q1(states)   
        q2 = self.q2(states)
        min_Q = torch.min(q1,q2)

        actor_loss = (action_probs * (alpha * log_pis - min_Q )).sum(1).mean()
        log_action_pi = torch.sum(log_pis * action_probs, dim=1)
        return actor_loss, log_action_pi

    def soft_update(self, local_model , target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def learn(self):

        states, actions, rewards, next_states, dones = self.memory.get_data()

        for _ in range(self.n_epochs):

            batches = self.memory.generate_batches()
            for batch in batches:
                
                batch_states = states[batch]
                batch_next_states = next_states[batch]

                with torch.no_grad():
                    _, action_probs, log_pis = self.policy.evaluate(batch_next_states)


                    Q_target1_next = self.target_q1(batch_next_states)
                    Q_target2_next = self.target_q2(batch_next_states)
                    Q_target_next = action_probs * (torch.min(Q_target1_next, Q_target2_next) - self.alpha * log_pis)

                    # Compute Q targets for current states (y_i)
                    Q_targets = (rewards[batch] + (self.gamma * (1 - dones[batch]) * Q_target_next.sum(dim=1))).unsqueeze(1)
                

                # Compute critic loss
                q1 = self.q1(batch_states).gather(1, actions[batch].long().unsqueeze(1))
                q2 = self.q2(batch_states).gather(1, actions[batch].long().unsqueeze(1))

                critic1_loss =  0.5 * F.mse_loss(q1, Q_targets)
                critic2_loss =  0.5 * F.mse_loss(q2, Q_targets)

                # Update critics
                # critic 1
                self.q1_optimizer.zero_grad()
                critic1_loss.backward(retain_graph=True)
                clip_grad_norm_(self.q1.parameters(), self.clip_grad_param)
                self.q1_optimizer.step()
                # critic 2
                self.q2_optimizer.zero_grad()
                critic2_loss.backward()
                clip_grad_norm_(self.q2.parameters(), self.clip_grad_param)
                self.q2_optimizer.step()

                current_alpha = copy.deepcopy(self.alpha)
                actor_loss, log_pis = self.update_policy(batch_states, current_alpha)
                self.policy_optimizer.zero_grad()
                actor_loss.backward()
                self.policy_optimizer.step()
      
                # Compute alpha loss
                alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
                self.log_alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.log_alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().detach()
              
                # ----------------------- update target networks ----------------------- #

                self.soft_update(self.q1, self.target_q1)
                self.soft_update(self.q2, self.target_q2)

                #rospy.loginfo("FINNN")


        #self.memory.clear_data()


        return critic1_loss, critic2_loss, actor_loss, self.alpha



        
    