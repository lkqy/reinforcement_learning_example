#coding=utf-8
import numpy as np
from mxnet.gluon import data as gdata
from mxnet import ndarray as nd
from mxnet import init
from mxnet import gluon
from mxnet import autograd
from mxnet.gluon import data as gdata
from mxnet.gluon import loss as gloss, nn


class DQN(object):
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

        self.nn = nn.Sequential()
        self.nn.add(nn.Dense(10, activation='relu'))
        self.nn.add(nn.Dense(10, activation='relu'))
        self.nn.add(nn.Dense(len(actions)))
        self.nn.initialize(init.Normal(sigma=0.01))
        self.trainer = gluon.Trainer(self.nn.collect_params(), 'sgd', {'learning_rate': 0.01})
        self.loss = gloss.L2Loss()

    def train(self, datas, labels, batch_size=3, epoch=10):
        dataset = gdata.ArrayDataset(datas, labels)
        data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
        for _ in range(epoch):
            for X, y in data_iter:
                with autograd.record():
                    res = self.nn(X)
                    l = self.loss(res, y)
                l.backward()
                self.trainer.step(batch_size)

            for X, y in data_iter:
                l = self.loss(self.nn(X), y)
                print 'loss', l
                break

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

if __name__ == '__main__':
    q = DQN(None, 0.2, 0.99, 0.15, [0, 1])
    datas = []
    labels = []
    for x in range(1, 100):
        for y in range(1, 100):
            _x = x/10.0
            _y = y/10.0
            datas.append(nd.array([_x, _y]))
            labels.append(nd.array([_x-(_y), (_x)+_y]))

    dataset = gdata.ArrayDataset(datas, labels)
    q.train(datas, labels, batch_size=100, epoch=100)
