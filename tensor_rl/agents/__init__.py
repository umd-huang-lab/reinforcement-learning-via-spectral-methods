'''
Implementations of standard RL agents:

	AgentClass: Contains the basic skeleton of an RL Agent.
	QLearningAgentClass: Q-Learning.
	LinearQAgentClass: Q-Learning with a Linear Approximator.
	RandomAgentClass: Random actor.
	RMaxAgentClass: R-Max.
	LinUCBAgentClass: Contextual Bandit Algorithm.
'''

# Grab agent classes.
from tensor_rl.agents.AgentClass import Agent
from tensor_rl.agents.FixedPolicyAgentClass import FixedPolicyAgent
from tensor_rl.agents.QLearningAgentClass import QLearningAgent
from tensor_rl.agents.DoubleQAgentClass import DoubleQAgent
from tensor_rl.agents.DelayedQAgentClass import DelayedQAgent
from tensor_rl.agents.RandomAgentClass import RandomAgent
from tensor_rl.agents.RMaxAgentClass import RMaxAgent
from tensor_rl.agents.OptimalAgentClass import OptimalAgent
try:
	from tensor_rl.agents.func_approx.DQNAgentClass import DQNAgent
except ImportError:
	print("Warning: Tensorflow not installed.")
	pass

from tensor_rl.agents.bandits.LinUCBAgentClass import LinUCBAgent