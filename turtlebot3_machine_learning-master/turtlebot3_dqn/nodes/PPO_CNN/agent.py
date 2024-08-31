import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import os
from nodes.PPO_CNN.networks import CNNActor, CNNCritic
from nodes.PPO_CNN.memory import Memory
import rospy

class PPO:
    def __init__(self, input_dims, fc1_dims = 256, fc2_dims = 256, n_actions = 5, gamma=0.99, alpha=0.0003,
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64, n_epochs=5, using_camera = 0):

        self.using_camera = using_camera
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = CNNActor(input_shape = input_dims, n_actions=n_actions, name='cnn_actor')
        self.critic = CNNCritic(input_shape = input_dims, name="critic")
        
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=alpha))
        
        self.memory = Memory(batch_size, self.using_camera)

    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_data(state, action, probs, vals, reward, done)

    def save_models(self):
        self.actor.save_weights(self.actor.save_directory)
        self.critic.save_weights(self.critic.save_directory)

    def load_models(self):
        self.actor.load_weights(self.actor.save_directory)
        self.critic.load_weights(self.critic.save_directory)

    def choose_action(self, observation):

        state = tf.convert_to_tensor([observation])

        probs = self.actor(state)
        dist = tfp.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state)

        action = action.numpy()[0]
        value = value.numpy()[0]
        log_prob = log_prob.numpy()[0]

        return action, log_prob, value

    def learn(self):

        actor_loss_mean = []
        critic_loss_mean = []

        for _ in range(self.n_epochs):
            state_arr, old_prob_arr, vals_arr, action_arr, reward_arr, dones_arr, batches = self.memory.generate_data()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t

            for batch in batches:
                states = tf.convert_to_tensor(state_arr[batch], dtype=tf.float32)
                old_probs = tf.convert_to_tensor(old_prob_arr[batch], dtype=tf.float32)
                actions = tf.convert_to_tensor(action_arr[batch], dtype=tf.int32)
                advantage_batch = tf.convert_to_tensor(advantage[batch], dtype=tf.float32)
                vals_arr_batch = tf.convert_to_tensor(vals_arr[batch], dtype=tf.float32)

                with tf.GradientTape(persistent=True) as tape:
                    probs = self.actor(states)
                    dist = tfp.distributions.Categorical(probs)
                    new_probs = dist.log_prob(actions)

                    critic_value = self.critic(states)
                    critic_value = tf.squeeze(critic_value, axis=1)

                    prob_ratio = tf.exp(new_probs - old_probs)
                    weighted_probs = advantage * prob_ratio

                    clipped_probs = tf.clip_by_value(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    weighted_clipped_probs = clipped_probs * advantage

                    actor_loss = -tf.reduce_mean(tf.minimum(weighted_probs, weighted_clipped_probs))

                    returns = advantage + vals_arr
                    critic_loss = tf.reduce_mean(tf.square(returns - critic_value))

                actor_params = self.actor.trainable_variables
                actor_grads = tape.gradient(actor_loss, actor_params)
                self.actor.optimizer.apply_gradients(zip(actor_grads, actor_params))

                critic_params = self.critic.trainable_variables
                critic_grads = tape.gradient(critic_loss, critic_params)
                self.critic.optimizer.apply_gradients(zip(critic_grads, critic_params))

                actor_loss_mean.append(actor_loss.numpy())
                critic_loss_mean.append(critic_loss.numpy())

        self.memory.clear_data()

        return np.array(actor_loss_mean).mean(), np.array(critic_loss_mean).mean()