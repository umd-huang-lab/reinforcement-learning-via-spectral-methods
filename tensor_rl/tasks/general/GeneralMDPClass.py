#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 20:27:33 2019

"""

# Python imports.
from __future__ import print_function
import random
import math
import sys, os
import scipy
import numpy as np
from collections import defaultdict

# Other imports.
from tensor_rl.mdp.MDPClass import MDP
from tensor_rl.tasks.general.GeneralStateClass import GeneralState

# Fix input to cooperate with python 2 and 3.
try:
   input = raw_input
except NameError:
   pass

class GeneralMDP(MDP):
    ''' Class for a general MDP '''

    # Static constants.
#    ACTIONS = [0, 1, 2]
#    N_ACTIONS = 3

    def __init__(self,
                 n_states,
                 n_actions,
                 tuples=[],
                 init_state = 0,
                 gamma = 0.5,
                 name="general",
                 par_tensor=None):
        self.n_actions = n_actions
        self.actions = list(range(n_actions))
        self.n_states = n_states
        init_state = GeneralState(init_state)
        
        MDP.__init__(self, self.actions, self._transition_func, self._reward_func, init_state=init_state, gamma=gamma)
        
        self.name = name
        self._init_states()
        
        self.par_tensor = self._init_par_tensor(tuples)
    
    def _init_par_tensor(self, tuples):
        '''tuples: [(state, action, next_state, probability, reward)]'''
        tensor = np.zeros((self.n_actions, self.n_states, self.n_states+1))
        for tup in tuples:
            s, a, s1, prob, r = tup
            tensor[a][s][s1] = prob
            tensor[a][s][-1] += r * prob
        return tensor
    

    def _init_states(self):
        self.states = []
        self.state_map = {}
        for i in range(self.n_states):
            self.states.append(GeneralState(i))
            self.state_map[GeneralState(i)] = i

    def _choose_by_dist(self, prob_vec):
        ''' return a chosen state number '''
        
        rand = random.random()
#        print("a random number:", rand)
#        print("prob_dist", prob_vec)
        s = 0
        for i in range(len(prob_vec)-1):
            s += prob_vec[i]
            if s > rand:
                return GeneralState(i)
#        print("nothing has been chosen!")
        return None    

    def _transition_func(self, state, action):
#        print("(", action, ")", end=" ")
        next_state = self._choose_by_dist(self.par_tensor[action][state.get_data()])
#        print("next", next_state, end=' ')
        return next_state
        
        
    def _reward_func(self, state, action, next_state):

        return self.par_tensor[action][state.get_data()][self.n_states]

    
    def __str__(self):
        return self.name
    
    
    
    
    
    

    