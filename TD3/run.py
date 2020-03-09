import numpy as np
import tensorflow as tf
import keras.backend as K

from td3 import TD3

import gym



def main(env_name='LunarLanderContinuous-v2',
         low_list=[-1, -1],
         high_list=[1, 1],
         pts_list=[1000,1000],
         modeltype='wolpertinger'):
    sess = tf.Session()
    K.set_session(sess)

    # Define environment
    # env = gym.make('Pendulum-v0')
    # env = gym.make('BipedalWalker-v2')
    env = gym.make(env_name)

    if modeltype == 'wolpertinger':
        ddpg = Wolpertinger(env, sess, low_list=low_list, high_list=high_list, points_list=pts_list)
    elif modeltype == 'ddpg':
        ddpg = DDPG(env, sess, low_action_bound_list=low_list, high_action_bound_list=high_list)
    elif modeltype == 'td3':
        ddpg = TD3(env, sess, low_action_bound_list=low_list, high_action_bound_list=high_list)

    # Main loop
    num_episodes = 200
    max_episode_len = 1000

    for i in range(num_episodes):
        total_reward = 0

        current_state = env.reset()

        for step in range(max_episode_len):
            current_state = current_state.reshape((1, ddpg.state_dim))
            action = ddpg.act(i, current_state)
            if ddpg.action_dim == 1:
                action = action.reshape((1, ddpg.action_dim))
            elif ddpg.action_dim > 1:
                action = action.reshape((1, ddpg.action_dim))[0]
                print('DEBUG ACTION: ', action)

            next_state, reward, done, info = env.step(action)
            next_state = next_state.reshape((1, ddpg.state_dim))
            total_reward += reward

            # if step == (max_episode_len-1):
            #     done = True
            #     print('Reward: ', total_reward)

            ddpg.train_critic()

            if (step % 2) == 0:
                # print('Training actor and updating weights at episode {} step {}'.format(i, step))
                ddpg.train_actor()
                ddpg.update_target_models()

            ddpg.replay_buffer.add(current_state, action, reward, next_state, done)
            current_state = next_state

            if done:
                break

        print('Total Reward for episode {}: {}'.format(i, total_reward))

        if i == (num_episodes-1):
            current_state = env.reset()
            for step in range(1000):
                env.render()
                current_state = current_state.reshape((1, ddpg.state_dim))
                action = ddpg.act(i, current_state)
                if ddpg.action_dim == 1:
                    action = action.reshape((1, ddpg.action_dim))
                elif ddpg.action_dim > 1:
                    action = action.reshape((1, ddpg.action_dim))[0]

                next_state, reward, done, info = env.step(action)
                next_state = next_state.reshape((1, ddpg.state_dim))

                current_state = next_state

                if done:
                    break



if __name__ == '__main__':
    main(env_name='Pendulum-v0', low_list=[-2], high_list=[2], pts_list=[1000], modeltype='td3')
    # main(env_name='LunarLanderContinuous-v2', low_list=[-1, -1], high_list=[1, 1], pts_list=[1000, 1000], modeltype='td3')
