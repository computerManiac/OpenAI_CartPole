import keras
from keras.models import Sequential,Model
from keras.layers import Dense
import numpy as np
import random
from keras.optimizers import Adam
from datetime import datetime
import gym
from collections import deque
import pickle

random.seed(datetime.now())

class DQN:

	def __init__(self,state_size,action_size):

		self.state_size = state_size
		self.action_size = action_size
		self.gamma = 0.9
		self.learning_rate = 0.01
		self.epsilon = 1
		self.epsilon_min = 0.001
		self.epsilon_decay = 0.995
		self.memory = deque(maxlen=2000)
		self.model = self._build_model()

	def _build_model(self):

		model = Sequential()
		model.add(Dense(24,input_dim=self.state_size,activation='relu'))
		model.add(Dense(24,activation='relu'))
		model.add(Dense(self.action_size,activation='linear'))
		model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))

		return model

	def remember(self,state,action,reward,next_state,done):
		self.memory.append((state,action,reward,next_state,done))

	def act(self,env,state):

		if (random.random() > self.epsilon):
			return np.argmax(self.model.predict(state)[0])

		return env.action_space.sample()

	def replay(self,batch_size=10):

		batch = random.sample(self.memory,batch_size)

		for state,action,reward,next_state,done in batch:

			target = reward

			if not done:
				target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

			target_value = self.model.predict(state)
			target_value[0][action] = target

			self.model.fit(state,target_value,epochs=1,verbose=0)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def save_model(self,file_name):

		hyper_params = {'epsilon':self.epsilon,'learning_rate':self.learning_rate,'gamma':self.gamma}
		with open('hyper_params.pickle','wb') as f:
			pickle.dump(hyper_params,f)
		self.model.save(file_name)


if __name__ == '__main__':

	env = gym.make('CartPole-v0')
	agent = DQN(4,env.action_space.n)
	#agent.model = load_model('dqn_model.h5')
	#hyper_params = pickle.load(open('hyper_params.pickle','rb'))

	max_episodes = int(input('Enter no. of training episodes: '))
	max_steps = 500

	for ep in range(max_episodes):

		state = env.reset()
		state = np.reshape(state,(1,4))
		total_rewards = 0

		for steps in range(max_steps):

			action = agent.act(env,state)

			new_state,reward,done,info = env.step(action)
			new_state = np.reshape(new_state, (1, 4))
			agent.remember(state,action,reward,new_state,done)
			total_rewards += reward

			state = new_state

			if done:
				break

		print('Episode {}/{} Steps:{} Score: {}'.format(ep+1,max_episodes,steps+1,total_rewards))
		agent.replay()

	agent.save_model('dqn_model.h5')

