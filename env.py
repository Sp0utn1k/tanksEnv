import numpy as np
from numpy import random
import math,time,sys,os
from collections import namedtuple

import torch
from torch import nn, optim
import cv2 as cv

device = "cuda"

class Environment:

	def __init__(self,args):
		self.N_red = args.N_red
		self.N_blue = args.N_blue
		self.obs_size = args.obs_size
		self.size  = args.size
		self.visibility = args.visibility
		self.R50 = args.R50
		self.alive = {}
		self.positions = {}
		self.N_episodes = args.N_episodes
		self.aim = {}
		self.red_agents = [Agent('red',args,id=i) for i in range(self.N_red)]
		self.blue_agents = [Agent('blue',args,id=i+self.N_red) for i in range(self.N_blue)]
		self.agents = self.red_agents+self.blue_agents
		self.all_agents = self.agents
		self.obstacles = args.obstacles
		if args.add_graphics:
			self.graphics = Graphics(args)
			self.plot = True
			self.twait = 1000.0/args.fps
		else:
			self.graphics = None
			self.plot = False
		self.reset()

	def reset(self):
		self.agents = self.all_agents
		self.red_agents = [agent for agent in self.agents if agent.team=="red"]
		self.blue_agents = [agent for agent in self.agents if agent.team=="blue"]
		self.target_list = {}
		for agent in self.agents:
			self.alive[agent] = True
			self.aim[agent] = None 
			self.positions[agent] = (random.randint(self.size),random.randint(self.size))

	def observation(self,a1,a2):
		vis = self.is_visible(a1,self.positions[a2])
		if vis:
			friend = a1.team == a2.team
			x2,y2 = self.positions[a2]
			x1,y1 = self.positions[a1]
			rel_pos = [x2-x1,y2-y1]

			# Normalize :
			rel_pos = np.array(rel_pos).astype(np.float64)
			rel_pos *= 1/self.visibility
			x,y = list(rel_pos)

			obs = [vis,friend,x,y]
			return obs
		return []

	def observations(self,a1):
		obs = []
		self.target_list[a1] = []
		for a2 in self.agents:
			if not a1 is a2:
				obsi = self.observation(a1,a2)
				if len([obsi]) > 0:
					obs += obsi
					self.target_list[a1] += [a2]

		assert len(obs) <= self.obs_size, "Too much observations"
		pad = [not i%4==0 for i in range(self.obs_size-len(obs))]
		obs += pad
		return obs

	def is_visible(self,a1,pos2):
		pos1 = self.positions[a1]
		dist = norm(pos1,pos2)
		if dist > self.visibility:
			return False
		LOS = los(pos1,pos2)
		for pos in LOS:
			if pos in self.obstacles:
				return False
			for agent in self.agents:
				if pos == self.positions[agent]:
					return False
		return True

	def is_valid_action(self,agent,act):
		act = agent.actions[act]
		if act == 'nothing':
			return True
		if act in ['up','down','left','right']:
			return self.is_free(self.next_tile(agent,act))
		if act == 'shoot':
			return self.aim[agent] in self.agents
		if 'aim' in act:
			target = int(act[3:])
			if target < len(self.target_list[agent]):
				return True
		return False

	def next_tile(self,agent,act):
		tile = self.positions[agent]
		x,y = tile
		assert act in ['up','down','left','right'], "Unauthorized movement"
		if act =='up':
			y += 1
		elif act == 'down':
			y -= 1
		elif act == 'left':
			x -= 1
		else:
			x += 1
		return [x,y]

	def is_free(self,tile):
		x,y = tile
		if x < 0 or x >= self.size or y < 0 or y >= self.size:
			return False
		return True

	def action(self,agent,action):
		if not self.is_valid_action(agent,action):
			return
		act = agent.actions[action]
		if act in ['up','down','left','right']:
			self.positions[agent] = self.next_tile(agent,act)
			# print(f'Agent {agent.id} ({agent.team}) goes {act}')
		if act == 'shoot':
			self.fire(agent)
			target = self.aim[agent]
			if self.plot:
				print(f'Agent {agent.id} ({agent.team}) shots at agent {target.id} ({target.team})')
		if 'aim' in act:
			target = int(act[3:])
			self.aim[agent] = self.target_list[agent][target]
			target = self.aim[agent]
			if self.plot:
				print(f'Agent {agent.id} ({agent.team}) aims at agent {target.id} ({target.team})')

	def fire(self,agent):
		target = self.aim[agent]
		distance = norm(self.positions[agent],self.positions[target])
		hit = random.rand() < self.Phit(distance)
		if hit:
			self.alive[target] = False
	
	def Phit(self,r):
		return sigmoid(self.R50-r,6/self.R50)

	def episode(self):
		for agent in self.agents:
			agent.episode = []
		while not self.episode_over():
			if self.plot:
				self.show_image()
				cv.imshow('image',env.graphics.image)
				cv.waitKey(round(self.twait))
			random.shuffle(self.agents)
			for agent in self.agents:
				obs = self.observations(agent)
				action = agent.act(obs)
				agent.to_episode(obs,action)
				self.action(agent,action)
				self.agents = [agent for agent in self.agents if self.alive[agent]]
				self.red_agents = [agent for agent in self.agents if agent.team=="red"]
				self.blue_agents = [agent for agent in self.agents if agent.team=="blue"]
		env.reset()
		cv.destroyAllWindows()
		for agent in self.agents:
			agent.to_batch()

	def episode_over(self):
		if len(self.blue_agents) == 0 or len(self.red_agents) == 0:
			return True
		return False

	def show_image(self):
		if not self.plot:
			return
		self.graphics.reset()
		for agent in self.agents:
			id = agent.id
			team = agent.team
			pos = self.positions[agent]
			self.graphics.add_agent(id,team,pos)
		for [x,y] in self.obstacles:
			self.graphics.set_obstacle(x,y)


class Agent:
	def __init__(self,team,args,id=0):
		self.id = id
		self.team = team
		self.actions = create_actions_set(args.N_aim)
		self.actions_size = len(self.actions)
		self.device = args.device
		self.create_model(args)
		self.episode = []
		self.batch = []

	def set_env(self,env):
		self.env = env

	def act(self,obs):
		actions = [action for action in self.actions.keys()]
		prob = self.model(torch.Tensor([obs]).to(self.device))
		return random.choice(actions,p=prob.cpu().detach().numpy()[0])

	def create_model(self,args):
		hidden = args.hidden
		N = len(hidden)
		layers = (nn.Linear(args.obs_size,hidden[0]),nn.ReLU())
		for i in range(1,N):
			layers += (nn.Linear(hidden[i-1],hidden[i]),nn.ReLU())

		layers += (nn.Linear(hidden[-1],self.actions_size),
					nn.Dropout(p=args.dropout),
					nn.Softmax(dim=1))
		self.model = nn.Sequential(*layers).to(args.device)
		self.loss_f = nn.MSELoss() 
		self.optimizer = optim.Adam(params=self.model.parameters(), lr=1e-3, betas=(0.9, 0.999))

	def to_episode(self,obs,action):
		action_label = [i == action for i in range(self.actions_size)]
		self.episode += [(obs,action_label)]

	def to_batch(self):
		self.batch += [self.episode]
		self.episode = []


class Graphics:
	def __init__(self,args):
		self.size = args.im_size
		self.szi = self.size/args.size
		self.background_color = self.to_color(args.background_color)
		self.red = self.to_color(args.red)
		self.blue = self.to_color(args.blue)
		self.obstacles_color = self.to_color(args.obstacles_color)
		self.reset()

	def reset(self):
		self.image = np.full((self.size,self.size,3),self.background_color,dtype=np.uint8)

	def to_color(self,color):
		(r,g,b) = color
		color = (b,g,r)
		return np.array(color,dtype=np.uint8)

	def pixels_in_coord(self,x,y):
		res = []
		szi = self.szi
		for j in range(round(szi*(x)),round(szi*(x+1))):
			for i in range(round(szi*(y)),round(szi*(y+1))):
					yield [i,j]

	def assign_value(self,x,y,val):
		for c in self.pixels_in_coord(x,y):
			self.image[c[0],c[1],:] = val

	def center(self,x,y):
		return (round(x*self.szi),round((y+1)*self.szi))

	def set_blue(self,x,y):
		self.assign_value(x,y,self.blue)

	def set_red(self,x,y):
		self.assign_value(x,y,self.red)

	def set_obstacle(self,x,y):
		self.assign_value(x,y,self.obstacles_color)

	def add_agent(self,id,team,pos):
		[x,y] = pos
		if team=='red':
			self.set_red(x,y)
		elif team=='blue':
			self.set_blue(x,y)
		cv.putText(self.image,f'{id}',self.center(x,y),cv.FONT_HERSHEY_SIMPLEX,0.6*self.szi/15,(0,0,0),2)

	def erase_tile(self,x,y):
		self.assign_value(x,y,self.background_color)

def norm(vect1,vect2):
	x1,y1 = vect1
	x2,y2 = vect2
	res = (x2-x1)**2 + (y2-y1)**2
	return math.sqrt(res)

def create_actions_set(N_aim):
	actions = {1:'up',2:'down',3:'left',4:'right'}
	actions[0] = 'nothing'
	actions[5] = 'shoot'
	for i in range(N_aim):
		actions[6+i] = f'aim{i}'
	return actions

def sigmoid(x,l):
	return 1.0/ (1+math.exp(-x*l))

def los(vect1,vect2):
	los_dict = {}
	[x1,y1] = vect1
	[x2,y2] = vect2
	N = round(norm(vect1,vect2))
	dy = (y2-y1)/N
	dx = (x2-x1)/N

	x_iter = [round(x1+i*dx) for i in range(N)]
	y_iter = [round(y1+i*dy) for i in range(N)]
	x = []
	y = []

	for (xi,yi) in zip(x_iter,y_iter):
		if (xi,yi) not in zip(x,y) and (xi,yi) != (x1,y1) and (xi,yi) != (x2,y2):
			yield [xi,yi]

	# LOS = [list(pos) for pos in zip(x,y)]
	# return LOS

if __name__ == "__main__":

	from setup import args
	env = Environment(args)
	env.episode()