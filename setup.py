class Args:
	# Environment
	size = 50
	N_blue = 3
	N_red = 2
	N_agents = N_red+N_blue
	visibility = 20.0
	R50 = 12.0
	N_episodes = 5
	obstacles = [[x,y] for x in range(22,29) for y in range(22,29)]

	# Agent
	N_aim = 10
	obs_size = 40
	dropout = 0.4
	hidden = [256,128,64]
	device = "cpu"

args = Args()