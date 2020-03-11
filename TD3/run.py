import numpy as np
import tensorflow as tf
from collections import deque
import keras.backend as K

from td3 import TD3

import gym



def main(env_name,
         low_list,
         high_list,
         pts_list):
    sess = tf.Session()
    K.set_session(sess)

    # Define environment
    env = gym.make(env_name)

    td3 = TD3(env, sess, low_action_bound_list=low_list, high_action_bound_list=high_list)

    # Main loop
    num_episodes = 200
    max_episode_len = 1000

    scores_deque = deque(maxlen=50)
    for i in range(num_episodes):
        total_reward = 0

        current_state = env.reset()

        for step in range(max_episode_len):
            current_state = current_state.reshape((1, td3.state_dim))
            action = td3.act(i, current_state)
            if td3.action_dim == 1:
                action = action.reshape((1, td3.action_dim))
            elif td3.action_dim > 1:
                action = action.reshape((1, td3.action_dim))[0]

            next_state, reward, done, info = env.step(action)
            next_state = next_state.reshape((1, td3.state_dim))
            total_reward += reward

            td3.train_critic()

            # Delayed training for policy
            if (step % 2) == 0:
                td3.train_actor()
                td3.update_target_models()

            td3.replay_buffer.add(current_state, action, reward, next_state, done)
            current_state = next_state

            if done:
                break

        scores_deque.append(total_reward)
        score_average = np.mean(scores_deque)

        print('Episode {}, Reward {}, Avg reward:{}'.format(i, total_reward, score_average))

        if score_average >= -300:

            td3.actor_model.save_weights('model_{}.h5'.format(env_name))

            # Display when finished
            current_state = env.reset()
            for step in range(1000):
                env.render()
                current_state = current_state.reshape((1, td3.state_dim))
                action = td3.act(i, current_state)
                if td3.action_dim == 1:
                    action = action.reshape((1, td3.action_dim))
                elif td3.action_dim > 1:
                    action = action.reshape((1, td3.action_dim))[0]

                next_state, reward, done, info = env.step(action)
                next_state = next_state.reshape((1, td3.state_dim))

                current_state = next_state

                if done:
                    break



if __name__ == '__main__':
    main(env_name='Pendulum-v0', low_list=[-2], high_list=[2], pts_list=[1000], modeltype='td3')
