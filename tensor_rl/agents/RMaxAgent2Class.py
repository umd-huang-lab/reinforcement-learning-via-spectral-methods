#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:19:28 2019

@author: yanchaosun
"""


# Python imports.
import random
import numpy as np
from collections import defaultdict

# Local classes.
from tensor_rl.agents.AgentClass import Agent

class RMaxAgent2(Agent):
    '''
    Implementation for the modified R-Max Agent [Sham's thesis]
    '''

    def __init__(self, states, state_map, actions, times,
                 gamma=0.95, horizon=3, s_a_threshold=2, 
                 name="RMax", greedy=False):
        name = name
        Agent.__init__(self, name=name, actions=actions, gamma=gamma)
        self.rmax = 1.0
        self.states = states
        self.state_map = state_map
        self.horizon = horizon
        self.s_a_threshold = s_a_threshold
        self.greedy = greedy
        self.reset()
        self.times = 0
        self.max_times = times
        
        s_len = len(self.states)
        shape = ((len(self.actions), s_len, s_len+1))
        self.par_tensor = np.zeros(shape)
        

    def reset(self):
        '''
        Summary:
            Resets the agent back to its tabula rasa config.
        '''
        self.rewards = defaultdict(lambda : defaultdict(list)) # S --> A --> [r_1, ...]
        self.transitions = defaultdict(lambda : defaultdict(lambda : defaultdict(int))) # S --> A --> S' --> counts
        self.r_s_a_counts = defaultdict(lambda : defaultdict(int)) # S --> A --> #rs
        self.t_s_a_counts = defaultdict(lambda : defaultdict(int)) # S --> A --> #ts
        self.prev_state = None
        self.prev_action = None
        
        self.norm_rewards = defaultdict(lambda : defaultdict(float))
        self.norm_transitions = defaultdict(lambda : defaultdict(lambda : defaultdict(float)))
        self.known_map = np.zeros((len(self.actions), len(self.states)))
        
        self.action_map = {}
        k = 0
        for a in self.actions:
            self.action_map[a] = k
            k += 1
        print(self.action_map)
        self.flag = False
        self.policy = defaultdict(type(self.actions[0]))
        self.q = np.zeros((len(self.states), len(self.actions)))

    def get_num_known_sa(self):
        return np.sum(self.known_map)

    def is_known(self, s, a):
        return 1 == self.known_map[self.action_map[a]][self.state_map[s]]
    
    def is_state_known(self, s):
        '''Whether a state is known (for all actions)'''
        return len(self.actions) == np.sum(self.known_map[:,self.state_map[s]])

    def act(self, state, reward):
            
        # Compute best action.
#        action = self.get_max_q_action(state)
        if self.flag:
            action = self.policy[state]
        else:
            self.times += 1
            self.update(self.prev_state, self.prev_action, reward, state)
            if self.is_state_known(state):
                if self.greedy:
                    action = self.get_best_action(state)
#                    action = self.get_policy_action(state, self.q)
                else:
#                    action = self.get_max_q_action(state)
                    action = self.get_policy_action(state, self.q)
            else:
                action = self.balanced_wander(state)

        # Update pointers.
        self.prev_action = action
        self.prev_state = state

        return action
    
    def dump_transitions(self):
        for a in self.actions:
            print("for action ", a)
            print("n", end=" ")
            for state in self.states:
                print(state, end=" ")
            print(" ")
            for state1 in self.states:
                print(state1, end=" ")
                for state2 in self.states:
                    print(self.transitions[state1][a][state2], end=" ")
                print(" ")
        print("known map")
        print(self.known_map)
        
    def dump_norm_trans(self):
        for a in self.actions:
            print("for action ", a)
            print("n", end=" ")
            for state in self.states:
                print(state, end=" ")
            print(" ")
            for state1 in self.states:
                print(state1, end=" ")
                for state2 in self.states:
                    print(self.norm_transitions[state1][a][state2], end=" ")
                print("reward ", self.norm_rewards[state1][a])
                print(" ")

    def balanced_wander(self, state):
        '''Return the action that has been taken for the least times'''
        min_action = random.choice(self.actions)
        min_visit_times = self.r_s_a_counts[state][min_action]
        
        for a in self.actions:
            if self.r_s_a_counts[state][a] < min_visit_times:
                min_visit_times = self.r_s_a_counts[state][a]
                min_action = a
        return min_action
    
    def print_policy(self):
        for s in self.states:
            print("s: ", s.get_data(), "a: ",self.policy[s])
    
    
    
    def update(self, state, action, reward, next_state):
        '''
        Args:
            state (State)
            action (str)
            reward (float)
            next_state (State)

        Summary:
            Updates T and R.
        '''
#        print("up:", state, action, next_state)
        if state != None and action != None:
            
            if self.r_s_a_counts[state][action] <= self.s_a_threshold:
                # Add new data points if we haven't seen this s-a enough.
                self.rewards[state][action] += [reward]
                self.r_s_a_counts[state][action] += 1
                self.norm_rewards[state][action] = sum(self.rewards[state][action]) / self.r_s_a_counts[state][action]

            if self.t_s_a_counts[state][action] <= self.s_a_threshold:
                self.transitions[state][action][next_state] += 1
                self.t_s_a_counts[state][action] += 1
                for s in self.transitions[state][action].keys():
                    self.norm_transitions[state][action][s] = self.transitions[state][action][s] / self.t_s_a_counts[state][action]
            
            if not self.is_known(state, action) and self.t_s_a_counts[state][action] >= self.s_a_threshold:
                self.known_map[self.action_map[action]][self.state_map[state]] = 1
                self.par_tensor = self.get_par_tensor()
#                print(self.known_map)
                if not self.greedy:
                    self.q = self.planning()
#        print("known num: ", self.get_num_known_sa(), end=" ")
        
        if self.times == self.max_times:
            self.par_tensor = self.get_par_tensor()
        if self.get_num_known_sa() >= len(self.states) * len(self.actions):
            print("TIMES: ", self.times)
            self.par_tensor = self.get_par_tensor()
            print(self.par_tensor)
            self.update_all()
    
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
        return q
    
    def update_all(self):
        '''
        After recovering parameters, we calculate the best actions 
        for all state-action pairs once and use them forever.
        '''
        q = self.planning()
        for s in self.state_map.keys():
#            print("getting policy for ", s)
            if self.greedy:
                self.policy[s] = self.get_best_action(s)
            else:
                self.policy[s] = self.get_policy_action(s, q)
        print("policies updated")
        self.flag = True
        self.print_policy()
    
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
            if rew > max_q:
                max_q = rew
                max_a = a
        return max_a
    
    def _compute_max_qval_action_pair(self, state, horizon=None):
        '''
        Args:
            state (State)
            horizon (int): Indicates the level of recursion depth for computing Q.

        Returns:
            (tuple) --> (float, str): where the float is the Qval, str is the action.
        '''
        # If this is the first call, use the default horizon.
        if horizon is None:
            horizon = self.horizon

        # Grab random initial action in case all equal
        best_action = random.choice(self.actions)
        max_q_val = self.get_q_value(state, best_action, horizon)

        # Find best action (action w/ current max predicted Q value)
        for action in self.actions:
            q_s_a = self.get_q_value(state, action, horizon)
            if q_s_a > max_q_val:
                max_q_val = q_s_a
                best_action = action

        return max_q_val, best_action

    def get_greedy_action(self, state):
        max_re = 0
        max_a = self.actions[0]
        for action in self.actions:
            if self.norm_rewards[state][action] > max_re:
                max_re = self.norm_rewards[state][action]
                max_a = action
        return max_a
            
    
    def get_max_q_action(self, state, horizon=None):
        '''
        Args:
            state (State)
            horizon (int): Indicates the level of recursion depth for computing Q.

        Returns:
            (str): The string associated with the action with highest Q value.
        '''

        # If this is the first call, use the default horizon.
        if horizon is None:
            horizon = self.horizon 
        return self._compute_max_qval_action_pair(state, horizon)[1]

    def get_max_q_value(self, state, horizon=None):
        '''
        Args:
            state (State)
            horizon (int): Indicates the level of recursion depth for computing Q.

        Returns:
            (float): The Q value of the best action in this state.
        '''

        # If this is the first call, use the default horizon.
        if horizon is None:
            horizon = self.horizon 
        return self._compute_max_qval_action_pair(state, horizon)[0]

    def get_q_value(self, state, action, horizon=None):
        '''
        Args:
            state (State)
            action (str)
            horizon (int): Indicates the level of recursion depth for computing Q.

        Returns:
            (float)
        '''

        # If this is the first call, use the default horizon.
        if horizon is None:
            horizon = self.horizon

        if horizon <= 0 or state.is_terminal():
            # If we're not looking any further.
            return self._get_reward(state, action)

        # Compute future return.
        expected_future_return = self.gamma*self._compute_exp_future_return(state, action, horizon)
        q_val = self._get_reward(state, action) + expected_future_return# self.q_func[(state, action)] = self._get_reward(state, action) + expected_future_return

        return q_val

    def _compute_exp_future_return(self, state, action, horizon=None):
        '''
        Args:
            state (State)
            action (str)
            horizon (int): Recursion depth to compute Q

        Return:
            (float): Discounted expected future return from applying @action in @state.
        '''

        # If this is the first call, use the default horizon.
        if horizon is None:
            horizon = self.horizon

#        next_state_dict = self.transitions[state][action]
#
#        denominator = float(sum(next_state_dict.values()))
#        state_weights = defaultdict(float)
#        for next_state in next_state_dict.keys():
#            count = next_state_dict[next_state]
#            state_weights[next_state] = (count / denominator)
        
#        weighted_future_returns = [self.get_max_q_value(next_state, horizon-1) * state_weights[next_state] for next_state in next_state_dict.keys()]

        weighted_future_returns = [self.get_max_q_value(next_state, horizon-1) * 
                                   self.norm_transitions[state][action][next_state] for next_state 
                                   in self.norm_transitions[state][action].keys()]

        return sum(weighted_future_returns)

    def _get_reward(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns:
            Believed reward of executing @action in @state. If R(s,a) is unknown
            for this s,a pair, return self.rmax. Otherwise, return the MLE.
        '''

#        if self.r_s_a_counts[state][action] >= self.s_a_threshold:
#            # Compute MLE if we've seen this s,a pair enough.
#            rewards_s_a = self.rewards[state][action]
#            return float(sum(rewards_s_a)) / len(rewards_s_a)
        
        if self.is_known(state, action):
            return self.norm_rewards[state][action]
#        if self.is_known(state, action):
#            if self.norm_rewards[state][action] > 0.5:
#                return 1.0
#            else:
#                return 0.0
        else:
            # Otherwise return rmax.
            return self.rmax

    def get_par_tensor(self):
        
        s_len = len(self.states)
        shape = ((len(self.actions), s_len, s_len+1))
        para_tensor = np.zeros(shape)

        for s1 in self.states:
            for a in self.actions:
#                for s2 in self.states:
#                    para_tensor[self.action_map[a]][self.state_map[s1]][self.state_map[s2]] = self.norm_transitions[s1][a][s2]
#                para_tensor[self.action_map[a]][self.state_map[s1]][s_len] = self.norm_rewards[s1][a]
                if self.is_known(s1,a):
                    for s2 in self.states:
                        para_tensor[self.action_map[a]][self.state_map[s1]][self.state_map[s2]] = self.norm_transitions[s1][a][s2]
                    para_tensor[self.action_map[a]][self.state_map[s1]][s_len] = self.norm_rewards[s1][a]
                else:
                    for s2 in self.states:
                        para_tensor[self.action_map[a]][self.state_map[s1]][self.state_map[s2]] = 1 if s1 == s2 else 0
                    para_tensor[self.action_map[a]][self.state_map[s1]][s_len] = self.rmax
                    
        return para_tensor