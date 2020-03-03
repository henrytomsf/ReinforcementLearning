import numpy as np
import pyflann
from ddpg import DDPG
import action_space



class Wolpertinger(DDPG):
    def __init__(self,
                 env,
                 sess,
                 low_list,
                 high_list,
                 points_list, # points in each dimension
                 k_ratio=0.1):
        super().__init__(env, sess)
        if self.continuous_action_space:
            self.action_space = action_space.Space(low_list, high_list, points_list)
            max_actions = self.action_space.get_num_actions()
        else:
            assert len(points_list)==1
            max_actions = points_list[0]
            self.action_space = action_space.Discrete(max_actions)
        
        self.knn = max(1, int(max_actions * k_ratio))

    def get_action_space(self):
        return self.action_space

    def act(self, state):
        proto_action = super().act(state)

        return self.wolpertinger_action(state, proto_action)
    
    def wolpertinger_action(self, state, proto_action):
        # get the proto action's knn
        actions = self.action_space.search_point(proto_action, self.knn)[0]

        # create all state action pairs from the search
        states = np.tile(state, [len(actions), 1])

        # eval each pair from critic
        actions_eval = self.critic_model.predict([states, actions])

        # find index of action pair with max Q value
        max_index = np.argmax(actions_eval)

        return actions[max_index]
