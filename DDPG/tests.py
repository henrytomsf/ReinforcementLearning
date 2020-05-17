import unittest
from collections import deque
import tensorflow as tf
import keras.backend as K

from ddpg import DDPG
import gym


def ddpg_test():
    sess = tf.Session()
    K.set_session(sess)

    env = gym.make('Pendulum-v0')

    ddpg = DDPG(env, sess, low_action_bound_list=[-2], high_action_bound_list=[2])

    total_reward = 0

    current_state = env.reset()

    # Main
    num_episodes = 2
    max_episode_len = 1

    scores_deque = deque(maxlen=50)
    for i in range(num_episodes):
        total_reward = 0
        current_state = current_state.reshape((1, ddpg.state_dim))
        action = ddpg.act(i, current_state)
        if ddpg.action_dim == 1:
            action = action.reshape((1, ddpg.action_dim))
        elif ddpg.action_dim > 1:
            action = action.reshape((1, ddpg.action_dim))[0]

    return action


class Testing(unittest.TestCase):
    def test_action(self):
        self.assertEqual(ddpg_test().shape, (1, 1))



if __name__ == '__main__':
    unittest.main()
