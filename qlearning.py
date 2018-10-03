#coding=utf-8
import numpy as np

class Qlearning(object):
    def __init__(self, digitize_fun, alpha, gamma, epsion, actions):
        self.digitize_fun = digitize_fun
        self.alpha = alpha
        self.gamma = gamma
        self.epsion = epsion
        self.actions = actions
        self.count = 0

        self.q_table = {}
        for i in range(256):
            self.q_table[i] = {0:0, 1:0}


    def get_actions(self, observation):
        self.count += 1
        state = self.digitize_fun(observation)
        if (self.epsion * 0.995 ** self.count) > np.random.uniform(0, 1):
            next_action = np.random.choice(self.actions)
        else:
            next_action = max(self.q_table[state].items(), key=lambda x:x[1])[0]
        return next_action

    def learn(self, state, action, observation, reward):
        state = self.digitize_fun(state)
        next_state = self.digitize_fun(observation)

        next_action = max(self.q_table[next_state].items(), key=lambda x:x[1])[0]

        next_v = self.q_table[next_state][next_action]
        cur_v = self.q_table[state][action]
        self.q_table[state][action] = (1-self.alpha) * cur_v + self.alpha * (reward + self.gamma * next_v)

        return next_action
