import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Concatenate
from keras.optimizers import Adam
from keras.losses import mse
import keras.backend as K

import tensorflow as tf



class Critic:
    def __init__(self,
                 state_dim,
                 action_dim,
                 learning_rate):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

    def create_critic_model(self):
        state_input = Input(shape=[self.state_dim])
        action_input = Input(shape=[self.action_dim])

        # define label layer
        target_q = Input(shape=[1])

        # first critic
        state1_h1 = Dense(500, activation='relu')(state_input)
        state1_h2 = Dense(200, activation='relu')(state1_h1)

        action1_h1 = Dense(500)(action_input)

        merged1= Concatenate()([state1_h2, action1_h1])
        merged1_h1 = Dense(200, activation='relu')(merged1)
        output1 = Dense(1, activation='linear')(merged1_h1)

        # second critic
        state2_h1 = Dense(500, activation='relu')(state_input)
        state2_h2 = Dense(200, activation='relu')(state2_h1)

        action2_h1 = Dense(500)(action_input)

        merged2 = Concatenate()([state2_h2, action2_h1])
        merged2_h1 = Dense(200, activation='relu')(merged2)
        output2 = Dense(1, activation='linear')(merged2_h1)

        model = Model(input=[state_input, action_input, target_q], output=[output1, output2])

        loss = K.mean(mse(output1, target_q) + mse(output2, target_q))
        model.add_loss(loss)

        return state_input, action_input, model
