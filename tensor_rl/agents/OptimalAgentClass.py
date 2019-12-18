#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 19:55:43 2019

"""

# Python imports.
import random
import numpy as np
from collections import defaultdict

# Local classes.
from tensor_rl.agents.AgentClass import Agent

class OptimalAgent(Agent):
    '''
    Implementation for the modified R-Max Agent [Sham's thesis]
    '''

    def __init__(self, states, state_map, actions, par_tensor, times,
                 gamma=0.95, horizon=2, name="Optimal", greedy=False):
        name = name
        Agent.__init__(self, name=name, actions=actions, gamma=gamma)

        self.states = states
        self.state_map = state_map
        self.horizon = horizon
        self.greedy = greedy
        self.times = times
        self.par_tensor = par_tensor
        self.reset()
#        print(self.states)
#        print(self.actions)
        print(self.par_tensor)
        self.policy = defaultdict(type(self.actions[0]))
        self.update_all()

    def reset(self):
        '''
        Summary:
            Resets the agent back to its tabula rasa config.
        '''
        self.action_map = {}
        k = 0
        for a in self.actions:
            self.action_map[a] = k
            k += 1
    
    def act(self, state, reward):
#        print(state)
        action = self.policy[state]

        return action
    
    def get_r(self, state, action):
        return self.par_tensor[self.action_map[action]][self.state_map[state]][len(self.states)]
    
    def get_next_r(self, state, action):
        probs = self.par_tensor[self.action_map[action]][self.state_map[state]][:len(self.states)]
        rs = np.zeros((len(self.states)))
        for s in self.states:
            rs[self.state_map[s]] = max(self.par_tensor[:,self.state_map[s],len(self.states)])
        return np.dot(probs, rs)
        
    def get_best_action(self, state):
        max_a = random.choice(self.actions)
        max_q = self.get_r(state, max_a) + self.gamma * self.get_next_r(state, max_a)
        for a in self.actions:
            r = self.get_r(state, a)
            nr = self.get_next_r(state, a)
            rew = r + self.gamma * nr
            print(a, r, nr)
            if rew > max_q:
                max_q = rew
                max_a = a
        return max_a
    
    def get_policy_action(self, state, q):
        return self.actions[np.argmax(q[self.state_map[state]])]
    
    def get_ns_dist(self, state, action):
        return self.par_tensor[self.action_map[action]][self.state_map[state]][:len(self.states)]
    
    def get_state_vals(self, q):
        print("max:", np.max(q, axis=1))
        s_vals = np.zeros(len(self.states))
        for s in self.states:
            s_vals[self.state_map[s]] = q[self.state_map[s]][self.action_map[self.get_policy_action(s,q)]]
        return s_vals
    
    def planning(self, n_iter=10000):
        q = np.zeros((len(self.states), len(self.actions)))
        prev_q = np.copy(q)
        for i in range(n_iter):
            for s in self.states:
                for a in self.actions:
                    q[self.state_map[s]][self.action_map[a]] = self.get_r(s,a) + self.gamma * np.dot(self.get_ns_dist(s,a), np.max(q, axis=1))
            if np.linalg.norm(q-prev_q) < 1e-3:
                print("iter for ", i, "times")
                break
            prev_q = np.copy(q)
        print(q)
        return q
    
    def update_all(self):
        '''
        After recovering parameters, we calculate the best actions 
        for all state-action pairs once and use them forever.
        '''
        q = self.planning()
        for s in self.states:
            print("getting policy for ", s)
#            self.policy[s] = self.get_best_action(s)
            self.policy[s] = self.get_policy_action(s, q)
        print("policies updated")
        self.print_policy()

    def print_policy(self):
        for s in self.states:
            print("s: ", s.get_data(), "a: ",self.policy[s])

if __name__ == "__main__":
    states = [1, 2, 3]
    state_map = {1: 0, 2: 1, 3: 2}
    actions = [0, 1]
    par_tensor = np.array([[[.2, .3, .5, 1],[.2, .3, .5, 0],[.2, .3, .5, -1]],
                         [[.3, .5, .2, 0.5],[.3, .5, .2, 0.7],[.3, .5, .2, -0.8]]])

    ag = OptimalAgent(states, state_map, actions, par_tensor, 100)
    
#    print(ag.get_best_action(2))
    
    q = ag.planning(100)
    print(q)
    print(ag.get_policy_action(2, q))
    
    
    
    
    
    
    
    