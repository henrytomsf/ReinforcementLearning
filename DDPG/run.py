import numpy as np
from collections import deque
import tensorflow as tf
import keras.backend as K

from ddpg import DDPG
from wolpertinger import Wolpertinger

import gym



def main(env_name,
         low_list,
         high_list,
         pts_list,
         b_wolpertinger):
    sess = tf.Session()
    K.set_session(sess)

    # Define environment
    env = gym.make(env_name)

    if b_wolpertinger:
        ddpg = Wolpertinger(env, sess, low_list=low_list, high_list=high_list, points_list=pts_list)
    else:
        ddpg = DDPG(env, sess, low_action_bound_list=low_list, high_action_bound_list=high_list)

    # Main loop
    num_episodes = 5000
    max_episode_len = 200

    scores_deque = deque(maxlen=50)
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

            next_state, reward, done, info = env.step(action)
            next_state = next_state.reshape((1, ddpg.state_dim))
            total_reward += reward
            # print('DEBUG ACTION REWARD: ', action, reward)
            
            ddpg.replay_buffer.add(current_state, action, reward, next_state, done)
            current_state = next_state

            if (step % 2) == 0:
                ddpg.train()
                ddpg.update_target_models()

            if done:
                break

        scores_deque.append(total_reward)
        score_average = np.mean(scores_deque)

        print('Episode {}, Reward {}, Avg reward:{}'.format(i, total_reward, score_average))
        

        if score_average >= -300:

            ddpg.actor_model.save_weights('model_{}.h5'.format(env_name))

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

            break



if __name__ == '__main__':
    main(env_name='Pendulum-v0', low_list=[-2], high_list=[2], pts_list=[1000], b_wolpertinger=False)

