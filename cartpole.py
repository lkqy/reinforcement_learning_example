#coding=utf-8
import gym
import numpy as np
from qlearning import Qlearning
from deep_qlearning import DQN


def digitize_fun(state):
    def bins(clip_min, clip_max, num):
        return np.linspace(clip_min, clip_max, num + 1)[1:-1]

    car_pos, car_v, pole_angle, pole_v = state
    result =  [np.digitize(car_pos, bins(-2.4, 2.4, 4)),
               np.digitize(car_v, bins(-3.0, 3.0, 4)),
               np.digitize(pole_angle, bins(-0.5, 0.5, 4)),
               np.digitize(pole_v, bins(-2.0, 2.0, 4))]
    x = sum([x*(4**i) for i, x in enumerate(result)])
    return x


q_f = DQN(digitize_fun, 0.2, 0.99, 0.15, [0, 1])

max_number_of_steps = 200   # 每一场游戏的最高得分

goal_average_steps = 195
num_consecutive_iterations = 100
last_time_steps = np.zeros(num_consecutive_iterations)  # 只存储最近100场的得分（可以理解为是一个容量为100的栈）

env = gym.make('CartPole-v0')
for episode in range(5000):
    observation = env.reset()   # 初始化本场游戏的环境
    episode_reward = 0
    for t in range(max_number_of_steps):
        action = q_f.get_actions(observation)
#        env.render()
        next_observation, reward, done, info = env.step(action)
        q_f.learn(observation, action, next_observation, -200 if done else reward)
        observation = next_observation
        episode_reward += reward
        if done:
            print('%d Episode finished after %f time steps / mean %f' % (episode, t + 1, last_time_steps.mean()))
            last_time_steps = np.hstack((last_time_steps[1:], [episode_reward]))    # 更新最近100场游戏的得分stack
            break
    # 如果最近100场平均得分高于195
    if (last_time_steps.mean() >= goal_average_steps):
        print('Episode %d train agent successfuly!' % episode)
        break
