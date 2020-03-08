import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Concatenate
from keras.optimizers import Adam
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
        state_h1 = Dense(500, activation='relu')(state_input)
        state_h2 = Dense(200, activation='relu')(state_h1)

        action_input = Input(shape=[self.action_dim])
        action_h1 = Dense(500)(action_input)

        merged = Concatenate()([state_h2, action_h1])
        merged_h1 = Dense(200, activation='relu')(merged)
        output = Dense(1, activation='linear')(merged_h1)
        model = Model(input=[state_input, action_input], output=output)

        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)

        return state_input, action_input, model
