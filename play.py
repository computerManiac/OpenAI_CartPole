import keras
from keras.models import Sequential,Model,load_model
from keras.layers import Dense
import numpy as np
import random
from keras.optimizers import Adam
from datetime import datetime
import gym
from collections import deque
from cartpole import *

env = gym.make('CartPole-v0')
agent = DQN(4,env.action_space.n)

test_episodes = int(input('Enter no. of test episodes: '))
agent.model = load_model('dqn_model.h5')
rewards = []

for ep in range(test_episodes):
	state = env.reset()
	total_rewards = 0
	state = np.reshape(state,(1,4))

	for step in range(300):

		env.render()
		action = np.argmax(agent.model.predict(state)[0])

		new_state,reward,done,info = env.step(action)
		new_state = np.reshape(new_state,(1,4))
		total_rewards += reward

		state = new_state

		if done:
			break

	print('Episode {}/{} Reward {} Steps {}'.format(ep+1,test_episodes,total_rewards,step+1))
	rewards.append(total_rewards)

print('Average Reward :',np.average(rewards))