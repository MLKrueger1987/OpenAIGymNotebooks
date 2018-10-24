import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy.random as rand
import numpy as np
import sympy
import random
"""
QLearning Update Rule

Q(s,a) = (1-a)Q(s,a) + a(r + g( Q(s',a))

In DQN the Q function is replaced with a ANN that approximates the Q
values of any given state action pair. Rather than querry a Q-Table that
contains the vaules for any given state, the learner can estimate the
value of a new state-action pair.

This gives the added benefit that we do not need to discretize the 
the state values. Discretizing inherently adds some bias as the designer
is making a determiniation about the appropriate cut-offs between any
two given values.

Caution is needed when using an ANN to approximate Q as the values can 
become unstable. This instability is due to the temporal proximity and
auto correlation between temorally adjacent states--state s+1 will be
simillar to s. To avoid this auto correlation, a replay memory can be 
used to decorrelate the experience used to train the agent.

A replay memory contains a finite number of experiences the agent has
encoutered which can be sampled for training. Using batch updates the
agent learns using random samples of the replay memory. This measure
enhances the stability of the training process.



States:
The state is the indicators and pricing info plus the current position.
So, if there are not shares of the stock owned, the position is 0 and 
the reward will always be 0. If there is a position in the stock, the 
reward is the daily return multiplied by the number of shares owned.
"""

class DQNLearner(object):
	"""docstring for QLearner"""
	def __init__(self,alpha, gamma,tau,state_vals,actions,init_eps = .95, 
		final_eps=.05, max_iter=1000, batch=128,memory_size=2000,
		max_decay=None,device='cpu'):
		self.device=torch.device('cpu')
		super(DQNLearner, self).__init__()
		if device =='cuda':
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.memory = Memory(memory_size,batch)
		self.Q = Network(state_vals,actions).to(self.device)
		self.Qprime = Network(state_vals,actions)
		self.Qprime.load_state_dict(self.Q.state_dict())
		self.Qprime.eval()
		self.Qprime = self.Qprime.to(self.device)
		self._optimizer = optim.Adam(params=self.Q.parameters(),lr=alpha)
		# self._optimizer = optim.RMSprop(params=self.Q.parameters())
		# self._loss = F.smooth_l1_loss
		self._loss = F.mse_loss
		self.num_a = actions
		self.num_s = state_vals
		self.eps = init_eps
		self.final_eps = final_eps
		if not max_decay == None:
			print('Using Max Decay')
			self.decay_period = max_decay
		else:
			self.decay_period = max_iter
		self.decay_rate = (init_eps - final_eps) / self.decay_period
		self.gamma = gamma
		self.lr_decay = optim.lr_scheduler.ExponentialLR(self._optimizer, .9993)
		self.tau = tau
		self.batch_count = 0
		self.cur_epoch = 0
		self.steps_done = 0
		self.eps_threshold = init_eps


	def set_eps(self, e, episode_count, min_e):
		self.min_eps = min_e
		self.max_eps = e
		self.episodes = episode_count
		self.epsilon = e
		x = sympy.symbols('x')
		decay_rate = sympy.solveset(sympy.Eq(self.min_eps + (self.max_eps - self.min_eps) * 
			sympy.exp(-1. * episode_count / x), min_e + .0000000000001), x,sympy.S.Reals)
		print (decay_rate)
		self.e_decay = next(iter(decay_rate))
		self.linear_decay = (self.max_eps / self.episodes)

	def query(self,s):
		
		if rand.random() < self.eps_threshold:
			# take a random action
			return torch.tensor(float(rand.randint(self.num_a)),dtype=torch.long)
		else:	
			with torch.no_grad():
				return torch.tensor(float(self.Q.forward(s.to(self.device)).argmax()),dtype=torch.long)

	def predict(self,s):
		with torch.no_grad():
			return torch.tensor(self.Q.forward(s.to(self.device)).argmax())

	def predict_batch(self,s):
		with torch.no_grad():
			return self.Q.forward(s).argmax(dim=1)

	def query_optimal(self,s):
		with torch.no_grad():
			return torch.tensor(float(self.Q.forward(s).argmax()),dtype=torch.long)

	def save_exp(self,s,a,r,sprime,done=None):
		self.memory.store((s,a,r,sprime,done))

	def learn(self,s,a,r,sprime,done=None):

# 		self.save_exp(s,a,r,sprime,done)
		if not self.memory.ready():
			return
		# Get a batch to perform learning on 
		batch = self.memory.get_batch()
		state,action,reward,sprime,done = zip(*batch)
		states = torch.cat(state).view(self.memory._batch_size,self.num_s).to(self.device)
		sprimes = torch.cat(sprime).view(self.memory._batch_size,self.num_s).to(self.device)
		actions = torch.tensor(action).view(self.memory._batch_size,1).to(self.device)
		rewards = torch.tensor(reward).view(self.memory._batch_size,1).to(self.device)
		done = torch.tensor(done).to(self.device)
		final_mask = done.ge(1)

		# calc the observed Q vals for the batch
		observed = self.Q.forward(states).gather(1,actions)

		# calc the discounted rewards from the next states based on the target
		discounted_fut_r = torch.zeros(self.memory._batch_size)
		discounted_fut_r = self.Qprime.forward(sprimes).detach().max(1)[0]



		# combine the fut rewards from target with cur rewards to get the exp.
		# Q val based on target
		expected = (rewards + (self.gamma * discounted_fut_r.float()).view(self.memory._batch_size,1))
		expected[final_mask] = rewards[final_mask]
		loss = self._loss(observed, expected)

		self.lossval = loss.item()

		self._optimizer.zero_grad()
		loss.backward()
		self._optimizer.step()


		if self.batch_count == 9:
			for paramA, paramC in zip(self.Qprime.parameters(),self.Q.parameters()):
				paramA.data.copy_(paramC.data * self.tau + paramA * (1-self.tau))

		self.batch_count = (self.batch_count + 1) % 10
		return self.lossval


	def next_epoch(self):
		# if self.decay_period < self.cur_epoch:
		# 	self.eps -= self.decay_rate
		self.lr_decay.step()
		# self.eps_threshold = self.final_eps + (self.eps - self.final_eps) * \
		# math.exp(-1. * self.steps_done / self.decay_period)
		# if self.steps_done < self.decay_period:
		# 	self.eps_threshold = self.eps - (self.decay_rate * self.steps_done)
		self.eps_threshold = self.final_eps + (self.eps - self.final_eps) * \
		math.exp(-1. * self.steps_done / self.decay_period)
		# self.eps_threshold *=
		self.steps_done += 1


	def set_memory(self, N, batch):
		self.memory = Memory(capacity=N,batch_size=batch)





class Network(nn.Module):
	"""docstring for Network"""
	def __init__(self,state_vals, actions):
		super(Network, self).__init__()
		self.il = nn.Linear(state_vals,100)
		self.hl1 = nn.Linear(100,100)
		self.hl2 = nn.Linear(100,100)
		self.ol = nn.Linear(100,actions)

	def forward(self,x):
		x = F.relu(self.il(x))
		x = F.relu(self.hl1(x))
		x = F.relu(self.hl2(x))
		x = self.ol(x)
		return x

# Transition = namedtuple('Transition',
#                         ('state', 'action', 'reward', 'sprime'))

class Memory(object):
	"""docstring for Memory"""
	def __init__(self, capacity, batch_size):
		super(Memory, self).__init__()
		self._mem = []
		self._pointer = 0
		self.capacity = capacity
		self._ready = False
		self._batch_size = batch_size

	def store(self,transition):
		if len(self._mem) < self.capacity:
			self._mem.append(None)
		if len(self._mem) >= self._batch_size*2:
			self._ready = True	
		self._mem[self._pointer] = transition
		self._pointer = (self._pointer + 1) % self.capacity
	
	def get_batch(self):
		# for k in range(len(k)-2):
		sample = random.sample(self._mem,self._batch_size)
		# sample.append(self._mem[self._pointer - 1])
			# ranks = torch.tensor([self.memory.index(s) for s in sample])
		return sample

	def clear(self):
		self._mem = []
		self._pointer = 0
		self._ready = False

	def ready(self):
		return self._ready

		