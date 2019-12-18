#!/usr/bin/env python

# Python imports.
import sys
import numpy as np
import math
import keras.backend as K
import tensorflow as tf

# Other imports.
import srl_example_setup
from tensor_rl.agents import QLearningAgent, RandomAgent, OptimalAgent, DelayedQAgent, DoubleQAgent
from tensor_rl.tasks import GridWorldMDP, GeneralMDP
from tensor_rl.run_experiments import run_agents_on_mdp, run_single_agent_on_mdp
from tensor_rl.tasks.low_rank.LowRankMDPClass import LowRankMDP
#from tensor_rl.tasks.four_room.FourRoomMDPClass import FourRoomMDP
from tensor_rl.agents.RMaxAgent2Class import RMaxAgent2
from tensor_rl.agents.RMaxTensorAgentClass import RMaxTensorAgent
from tensor_rl.utils import chart_utils
from optspace import opt_space, const_mask, max_diff, fro_diff

import warnings
warnings.filterwarnings("ignore")


def lowrank_test():
    """The synthetic task"""
    
    n = 20
    mdp = LowRankMDP(n, 10, rank = 2)

    ql_agent = QLearningAgent(actions=mdp.get_actions(), gamma=0.95)
    delay_ql_agent = DelayedQAgent(actions=mdp.get_actions(), gamma=0.95)
    rand_agent = RandomAgent(actions=mdp.get_actions())
    double_q_agent = DoubleQAgent(actions=mdp.get_actions())
    
    opt_agent = OptimalAgent(states=mdp.states, state_map=mdp.state_map, actions=mdp.get_actions(), 
                            par_tensor=mdp.par_tensor, times=1500)
#    
    rmax_agent = RMaxAgent2(states=mdp.states, state_map=mdp.state_map, actions=mdp.get_actions(), 
                            times=30000, s_a_threshold=50, greedy=False)
#    
    gim_agent = RMaxTensorAgent(states=mdp.states, state_map=mdp.state_map, actions=mdp.get_actions(),
                                   s_a_threshold=50, use_tensor=True, rank=10, mu=0.03, rho=0.8, 
                                   greedy=True, strict=False, origin_tensor=mdp.par_tensor, name="GIM")
#   
    run_agents_on_mdp([opt_agent, 
                       rmax_agent, gim_agent,
                       rand_agent, ql_agent, delay_ql_agent, double_q_agent], 
                      mdp, instances=1, episodes=10000, steps=10, open_plot=True, cumulative_plot=True)

  
    
def grid_test():
    """The GridWorld task"""
    
    n = 4
    mdp = GridWorldMDP(width=4, height=4, goal_locs=[(2, 2), (3, 3)], rand_init=True,
                       gamma=0.95, slip_prob=0.4, step_cost=0.2)
    origin_tensor = mdp.get_par_tensor()

#    
    ql_agent = QLearningAgent(actions=mdp.get_actions(), gamma=0.95)
    delay_ql_agent = DelayedQAgent(actions=mdp.get_actions(), gamma=0.95)
    rand_agent = RandomAgent(actions=mdp.get_actions())
    double_q_agent = DoubleQAgent(actions=mdp.get_actions())
    opt_agent = OptimalAgent(states=mdp.states, state_map=mdp.state_map, actions=mdp.get_actions(), 
                            par_tensor=origin_tensor, times=100000)
    
    rmax_agent = RMaxAgent2(states=mdp.states, state_map=mdp.state_map, actions=mdp.get_actions(), 
                            times=100000, s_a_threshold=20, greedy=False)
    
    gim_agent = RMaxTensorAgent(states=mdp.states, state_map=mdp.state_map, actions=mdp.get_actions(),
                                   s_a_threshold=20, use_tensor=True, rank=5, mu=0.03, rho=0.8, 
                                   greedy=False, origin_tensor=origin_tensor, name="GIM")
    run_agents_on_mdp([opt_agent, 
                       rmax_agent, gim_agent,
                       rand_agent, ql_agent, delay_ql_agent, double_q_agent], 
                      mdp, instances=1, episodes=10000, steps=10, open_plot=True, cumulative_plot=True)

    
def casinoland_test():
    """The CasinoLand task"""

    num_s = 8
    num_n = 4
    trans = [(0,0,0,1,0), (0,1,4,1,0), (0,2,1,1,0), (0,3,0,1,0),
           (1,0,1,1,0), (1,1,5,1,0), (1,2,2,1,0), (1,3,0,1,0),
           (2,0,2,1,0), (2,1,6,1,0), (2,2,3,1,0), (2,3,1,1,0),
           (3,0,3,1,0), (3,1,7,1,0), (3,2,3,1,0), (3,3,2,1,0),
           (4,0,0,1,0), (4,1,4,0.6,0), (4,1,0,0.4,198.75), (4,2,4,1,-100), (4,3,4,1,0),
           (5,0,1,1,0), (5,1,5,0.5,0), (5,1,1,0.5,160), (5,2,5,1,-100), (5,3,5,1,0),
           (6,0,2,1,0), (6,1,6,0.8,0), (6,1,2,0.2,500), (6,2,6,1,-100), (6,3,6,1,0),
           (7,0,3,1,0), (7,1,7,0.1,0), (7,1,3,0.9,72.22), (7,2,7,1,-100), (7,3,7,1,0)]
    mdp = GeneralMDP(n_states=num_s, n_actions=num_n, tuples=trans, name="CasinoLand")
    
    origin_tensor = mdp.par_tensor
    ql_agent = QLearningAgent(actions=mdp.get_actions(), gamma=0.95)
    delay_ql_agent = DelayedQAgent(actions=mdp.get_actions(), gamma=0.95)
    rand_agent = RandomAgent(actions=mdp.get_actions())
    double_q_agent = DoubleQAgent(actions=mdp.get_actions())
    opt_agent = OptimalAgent(states=mdp.states, state_map=mdp.state_map, actions=mdp.get_actions(), 
                             par_tensor=origin_tensor, times=1500)
    rmax_agent = RMaxAgent2(states=mdp.states, state_map=mdp.state_map, actions=mdp.get_actions(), 
                             times=10000, s_a_threshold=10, greedy=False)
    gim_agent = RMaxTensorAgent(states=mdp.states, state_map=mdp.state_map, actions=mdp.get_actions(),
                                    s_a_threshold=10, use_tensor=True, rank=10, mu=0.03, rho=0.8, 
                                    greedy=False, name="GIM")

    run_agents_on_mdp([opt_agent, 
                       rmax_agent, gim_agent,
                       rand_agent, ql_agent, delay_ql_agent, double_q_agent], mdp, 
                      instances=1, episodes=5000, steps=10, open_plot=True,
                      cumulative_plot=True)

def riverswim_test():
    """The RiverSwim task"""
    
    num_s = 6
    num_n = 2
    trans = [(0,0,0,1,5), (0,1,0,0.7,0), (0,1,1,0.3,0), 
             (1,0,0,1,0), (1,1,0,0.1,0), (1,1,1,0.6,0), (1,1,2,0.3,0), 
             (2,0,1,1,0), (2,1,1,0.1,0), (2,1,2,0.6,0), (2,1,3,0.3,0),
             (3,0,2,1,0), (3,1,2,0.1,0), (3,1,3,0.6,0), (3,1,4,0.3,0),
             (4,0,3,1,0), (4,1,3,0.1,0), (4,1,4,0.6,0), (4,1,5,0.3,0),
             (5,0,4,1,0), (5,1,4,0.7,0), (5,1,5,0.3,10000)]
    mdp = GeneralMDP(n_states=num_s, n_actions=num_n, tuples=trans, init_state=2, name="RiverSwim")

    
    origin_tensor = mdp.par_tensor
    ql_agent = QLearningAgent(actions=mdp.get_actions(), gamma=0.95)
    delay_ql_agent = DelayedQAgent(actions=mdp.get_actions(), gamma=0.95)
    rand_agent = RandomAgent(actions=mdp.get_actions())
    double_q_agent = DoubleQAgent(actions=mdp.get_actions())
    opt_agent = OptimalAgent(states=mdp.states, state_map=mdp.state_map, actions=mdp.get_actions(), 
                             par_tensor=origin_tensor, times=1500)
    rmax_agent = RMaxAgent2(states=mdp.states, state_map=mdp.state_map, actions=mdp.get_actions(), 
                             times=10000, s_a_threshold=50, greedy=True)
    gim_agent = RMaxTensorAgent(states=mdp.states, state_map=mdp.state_map, actions=mdp.get_actions(),
                                    s_a_threshold=50, use_tensor=True, rank=10, mu=0.03, rho=0.8, 
                                    greedy=False, name="GIM")
    
    run_agents_on_mdp([opt_agent, 
                       rmax_agent, gim_agent,
                       rand_agent, ql_agent, delay_ql_agent, double_q_agent], mdp, 
                      instances=1, episodes=5000, steps=20, open_plot=True,
                      cumulative_plot=True)


if __name__ == "__main__":
#    lowrank_test()   # test the synthetic task
#    grid_test()    # test the gridworld task
#    casinoland_test()     # test the casinoland task
    riverswim_test()     # test the riverswim task
