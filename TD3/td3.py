import numpy as np

from actor import Actor
from critic import Critic
from replaybuffer import ReplayBuffer

import tensorflow as tf
from keras.optimizers import Adam



class TD3:
    def __init__(self,
                 env,
                 sess,
                 low_action_bound_list,
                 high_action_bound_list):
        self.env = env
        self.sess = sess
        self.low_action_bound_list = low_action_bound_list # depends on the env
        self.high_action_bound_list = high_action_bound_list
        self.action_range_bound = [hi-lo for hi,lo in zip(self.high_action_bound_list, self.low_action_bound_list)]
        self.learning_rate = 0.0001
        self.exploration_noise = 0.1
        self.gamma = 0.90
        self.tau = 0.01
        self.buffer_size = 10000
        self.batch_size = 128
        self.policy_noise = 0.1
        self.noise_clip = 0.05
        self.exploration_episodes = 10
        # self.policy_freq = 2

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = len(self.low_action_bound_list) #self.env.action_space, make this into input
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

        # Creating FIRST CRITIC model, this is the one we train/optimize against
        critic_ = Critic(self.state_dim, self.action_dim, self.learning_rate)
        self.critic_state_input, self.critic_action_input, self.critic_model = critic_.create_critic_model()
        self.critic_model.compile(optimizer=Adam(lr=critic_.learning_rate), loss='')

        _, _, self.target_critic_model = critic_.create_critic_model()
        self.target_critic_model.compile(optimizer=Adam(lr=critic_.learning_rate), loss='')

        self.critic_grads = tf.gradients(self.critic_model.output[0], self.critic_action_input)

        self.sess.run(tf.initialize_all_variables())

    def __repr__(self):
        return 'TD3_gamma{}_tau{}'.format(self.gamma, self.tau)

    # TRAINING FUNCTIONS
    def train_actor(self):
        if self.replay_buffer.size() > self.batch_size:
            samples = self.replay_buffer.sample_batch(self.batch_size)

            current_states, actions, rewards, next_states, dones = samples

            predicted_actions = self.actor_model.predict(current_states)*self.high_action_bound_list #TODO create linear mapping for affine space

            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input: current_states,
                self.critic_action_input: predicted_actions
            })[0]

            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: current_states,
                self.actor_critic_grad: grads
            })

    def train_critic(self):
        if self.replay_buffer.size() > self.batch_size:
            samples = self.replay_buffer.sample_batch(self.batch_size)

            current_states, actions, rewards, next_states, dones = samples

            target_actions = self.target_actor_model.predict(next_states)*self.high_action_bound_list

            # CCOMPUTING FIRST CRITIC
            # introduce area of noise to action for smoothing purposes
            noise = np.random.normal(size=len(self.action_range_bound)) * self.policy_noise
            clipped_noise = np.clip(noise, -self.noise_clip, self.noise_clip)

            # added above noise to target_actions and clip to be in range of valid actions
            target_actions = np.clip((target_actions + clipped_noise), self.low_action_bound_list, self.high_action_bound_list)
            target_q1_values, target_q2_values = self.target_critic_model.predict([next_states, target_actions, np.random.rand(self.batch_size, 1)])

            target_q_values = np.minimum(target_q1_values, target_q2_values)

            target_q = rewards + self.gamma * target_q_values * (1-dones)

            # current_q1, current_q2 = self.critic_model.predict([current_states, actions, np.random.rand(self.batch_size, 1)])

            history = self.critic_model.fit([current_states, actions, target_q], verbose=0)
            # print('Loss: ',history.history['loss'])

    def train(self):
        if self.replay_buffer.size() > self.batch_size:
            samples = self.replay_buffer.sample_batch(self.batch_size)
            self.train_actor()
            self.train_critic()

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
            current_epsiode,
            current_state):
        if current_epsiode < self.exploration_episodes:
            return np.random.uniform(self.low_action_bound_list, self.high_action_bound_list)*self.high_action_bound_list
        else:
            action = self.actor_model.predict(current_state)*self.high_action_bound_list + np.random.normal(0, [self.exploration_noise*hi for hi in self.high_action_bound_list])
            return np.clip(action, self.low_action_bound_list, self.high_action_bound_list)
