import numpy as np

from actor import Actor
from critic import Critic
from replaybuffer import ReplayBuffer

import tensorflow as tf



class DDPG:
    def __init__(self,
                 env,
                 sess):
        self.env = env
        self.sess = sess
        self.action_bound = 2.0 # depends on the env
        self.learning_rate = 0.0001
        self.epsilon = 0.9
        self.epsilon_decay = 0.99995
        self.gamma = 0.90
        self.tau = 0.01
        self.buffer_size = 100000
        self.batch_size = 128

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = 1 #self.env.action_space, make this into input
        self.continuous_action_space = True

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        # Creating ACTOR model
        actor_ = Actor(self.state_dim, self.action_dim, self.learning_rate)
        self.actor_state_input, self.actor_model = actor_.create_actor_model()
        _, self.target_actor_model = actor_.create_actor_model()

        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.action_dim])

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights, -self.actor_critic_grad)

        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        # Creating CRITIC model
        critic_ = Critic(self.state_dim, self.action_dim, self.learning_rate)
        self.critic_state_input, self.critic_action_input, self.critic_model = critic_.create_critic_model()
        _, _, self.target_critic_model = critic_.create_critic_model()

        self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input)

        self.sess.run(tf.initialize_all_variables())

    # TRAINING FUNCTIONS
    def train_actor(self,
                    samples):
        current_states, actions, rewards, next_states, dones = samples
        
        predicted_actions = self.actor_model.predict(current_states)

        grads = self.sess.run(self.critic_grads, feed_dict={
            self.critic_state_input: current_states,
            self.critic_action_input: predicted_actions
        })[0]

        self.sess.run(self.optimize, feed_dict={
            self.actor_state_input: current_states,
            self.actor_critic_grad: grads
        })

    def train_critic(self,
                     samples):
        current_states, actions, rewards, next_states, dones = samples

        target_actions = self.target_actor_model.predict(next_states)
        target_q_values = self.target_critic_model.predict([next_states, target_actions])

        rewards = rewards + self.gamma * target_q_values * (1-dones)

        evaluation = self.critic_model.fit([current_states, actions], rewards, verbose=0)
        # print('Evaluation: ', evaluation)
    
    def train(self):
        if self.replay_buffer.size() > self.batch_size:
            samples = self.replay_buffer.sample_batch(self.batch_size)
            self.train_actor(samples)
            self.train_critic(samples)
    
    # TARGET MODEL UPDATES
    def update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        target_actor_model_weights = self.target_actor_model.get_weights()

        for i in range(len(target_actor_model_weights)):
            target_actor_model_weights[i] = actor_model_weights[i]*self.tau + target_actor_model_weights[i]*(1.0-self.tau)
        self.target_actor_model.set_weights(target_actor_model_weights)

    def update_critic_target(self):
        critic_model_weights = self.critic_model.get_weights()
        target_critic_model_weights = self.target_critic_model.get_weights()

        for i in range(len(target_critic_model_weights)):
            target_critic_model_weights[i] = critic_model_weights[i]*self.tau + target_critic_model_weights[i]*(1.0-self.tau)
        self.target_critic_model.set_weights(target_critic_model_weights)
    
    def update_target_models(self):
        self.update_actor_target()
        self.update_critic_target()

    # ACTING FUNCTION with epsilon greedy
    def act(self,
            current_state):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.actor_model.predict(current_state)*self.action_bound + np.random.normal() # TODO make action_bound general for all dimensions
        else:
            return self.actor_model.predict(current_state)*self.action_bound
