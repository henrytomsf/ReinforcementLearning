from collections import deque
import random
import numpy as np



class ReplayBuffer:
    def __init__(self,
                 buffer_size,
                 random_seed=123):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self,
            state,
            action,
            reward,
            next_state,
            done):
        experience = (state, action, reward, next_state, done)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
        
    def size(self):
        return self.count

    def sample_batch(self,
                     batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        
        batch = np.array(batch)
        
        current_states = np.stack(batch[:,0]).reshape((batch.shape[0],-1))
        actions = np.stack(batch[:,1]).reshape((batch.shape[0],-1))
        rewards = np.stack(batch[:,2]).reshape((batch.shape[0],-1))
        next_states = np.stack(batch[:,3]).reshape((batch.shape[0],-1))
        dones = np.stack(batch[:,4]).reshape((batch.shape[0],-1))

        return current_states, actions, rewards, next_states, dones

    def clear(self):
        self.buffer.clear()
        self.count = 0
