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

