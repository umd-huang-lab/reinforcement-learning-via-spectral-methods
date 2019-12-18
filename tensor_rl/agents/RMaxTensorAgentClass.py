#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:19:28 2019

@author: yanchaosun
"""


# Python imports.
import random
import math
from collections import defaultdict
import numpy as np
#import tensorly as tl
#from tensorly.decomposition import parafac
import tensorflow as tf
from matrix_completion import svt_solve, pmf_solve, calc_unobserved_rmse
#import t3f
from optspace import opt_space

tf.set_random_seed(0)
np.random.seed(0)

# Local classes.
from tensor_rl.agents.AgentClass import Agent

class RMaxTensorAgent(Agent):
    '''
    Implementation for tensor-assist RMax
    '''

    def __init__(self, states, state_map, actions, use_tensor=True, rank=2, mu=0.1,
                 gamma=0.95, horizon=3, s_a_threshold=2, rho=0.7, beta=0.2, name="tensor", 
                 greedy=True, strict=True, origin_tensor=None, os=False):
        name = name
        Agent.__init__(self, name=name, actions=actions, gamma=gamma)
        self.rmax = 1.0
        self.horizon = horizon
        self.states = states
        self.state_map = state_map
        self.s_a_threshold = s_a_threshold
        self.use_tensor = use_tensor
        self.rank = rank
        self.mu = mu
        self.greedy = greedy
        self.strict = strict
        self.rho = rho
        self.beta = beta
        self.reset()
        self.times = 0
        self.origin_tensor = origin_tensor
        self.os = os

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
        
#        self.norm_rewards = defaultdict(lambda : defaultdict(float))
#        self.norm_transitions = defaultdict(lambda : defaultdict(lambda : defaultdict(float)))
        self.known_map = np.zeros((len(self.actions), len(self.states)))
        
        s_len = len(self.states)
        shape = ((len(self.actions), s_len, s_len+1))
        self.par_tensor = np.zeros(shape)
        
        self.action_map = {}
        k = 0
        for a in self.actions:
            self.action_map[a] = k
            k += 1
        print(self.action_map)
        
        self.flag = False
        self.policy = defaultdict(type(self.actions[0]))

    def get_num_known_sa(self):
        return np.sum(self.known_map)

    def is_known(self, s, a):
        return 1 == self.known_map[self.action_map[a]][self.state_map[s]]
    
    def is_state_known(self, s):
        '''Whether a state is known (for all actions)'''
        return len(self.actions) == np.sum(self.known_map[:,self.state_map[s]])

    def act(self, state, reward):
#        self.dump_transitions()
    
        if self.flag:
            action = self.policy[state] #self.get_max_q_action(state)
#            print("don't update, take action", action, " in state ", state)
        
        else:
            self.times += 1
            if self.times % 20000 == 0:
#                self.dump_norm_trans()
                print(self.known_map)
                
            action = self.beta_greedy_walking(state)
#            if self.is_state_known(state):
#                if self.greedy:
#                    action = self.get_greedy_action(state)
#                else:
#                    action = self.get_max_q_action(state)
#            else:
#                action = self.beta_greedy_walking(state)
                
            # Update given s, a, r, s' : self.prev_state, self.prev_action, reward, state
            self.update(self.prev_state, self.prev_action, reward, state)
            
            # Update pointers.
            self.prev_action = action
            self.prev_state = state
                  
        return action

    def balanced_wander(self, state):
        '''Return the action that has been taken for the least times'''
        min_action = random.choice(self.actions)
        min_visit_times = self.r_s_a_counts[state][min_action]
        
        for a in self.actions:
            if self.r_s_a_counts[state][a] < min_visit_times:
                min_visit_times = self.r_s_a_counts[state][a]
                min_action = a
        return min_action

    def beta_greedy_walking(self, state):
        max_action = random.choice(self.actions)
        if random.random() < self.beta:
            return max_action

        known_num = np.sum(self.known_map[:,self.state_map[state]])

        '''if the current state is rho-known, return an action that goes to the least known state'''
        if known_num >= self.rho * len(self.actions):
            inv_known_map = np.ones((len(self.actions), len(self.states))) - self.known_map
            max_score = self.par_tensor[self.action_map[max_action]][self.state_map[state]][:-1].dot(inv_known_map[self.action_map[max_action]])
            for a in self.actions:
                score = self.par_tensor[self.action_map[a]][self.state_map[state]][:-1].dot(inv_known_map[self.action_map[a]])
                if score > max_score:
                    max_action = a
                    max_score = score
#            for s in self.states:
#                if np.sum(self.known_map[:,self.state_map[s]]) < self.rho * len(self.actions):
#                    for a in self.actions:
#                        if self.norm_transitions[state][a][s] > 0:
#                            return a
                    
#            return self.balanced_wander(state)
        
        else:
            max_visit_times = self.r_s_a_counts[state][max_action]
            for a in self.actions:
                if not self.is_known(state, a) and self.r_s_a_counts[state][a] > max_visit_times:
                    max_visit_times = self.r_s_a_counts[state][a]
                    max_action = a
                elif self.r_s_a_counts[state][a] == max_visit_times:
                    if np.sum(self.known_map[self.action_map[a]]) < np.sum(self.known_map[self.action_map[max_action]]):
#                    if random.random() > 0.5:
                        max_visit_times = self.r_s_a_counts[state][a]
                        max_action = a
        return max_action
            
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
                print("reward ", self.norm_rewards[state1][a])
                print(" ")
    
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
#        print("from ", state, ", take ", action, ", to", next_state)
        if state != None and action != None:
            s_id = self.state_map[state]
            a_id = self.action_map[action]
            
            if self.r_s_a_counts[state][action] <= self.s_a_threshold:
                # Add new data points if we haven't seen this s-a enough.
                self.rewards[state][action] += [reward]
                self.r_s_a_counts[state][action] += 1
                norm_reward = float(sum(self.rewards[state][action])) / self.r_s_a_counts[state][action] #max(float(sum(self.rewards[state][action])) / self.r_s_a_counts[state][action], 0)
                self.par_tensor[a_id][s_id][-1] = norm_reward
            
            if self.t_s_a_counts[state][action] <= self.s_a_threshold:
                self.transitions[state][action][next_state] += 1
                self.t_s_a_counts[state][action] += 1
                for s in self.transitions[state][action].keys():
                    norm_trans = self.transitions[state][action][s] / self.t_s_a_counts[state][action]
                    ss_id = self.state_map[s]
                    self.par_tensor[a_id][s_id][ss_id] = norm_trans
            
            if self.t_s_a_counts[state][action] >= self.s_a_threshold:
                self.known_map[self.action_map[action]][self.state_map[state]] = 1
                
#        print("known num: ", self.get_num_known_sa(), end=" ")
        if self.use_tensor:
            known_rate = 1.0 * self.get_num_known_sa() / (len(self.states) * len(self.actions))
#            if not self.flag and known_rate >= self.rho:
            
#            if known_rate >= self.rho: 
            if known_rate >= self.rho and self.is_rho_known():
                print("TIMES: ", self.times)
                self.complete_paras()

    def is_rho_known(self):
        if not self.strict:
            return True
        
        for s in self.states:
            known_num = 0
            for a in self.actions:
                if self.is_known(s,a):
                    known_num += 1
            if known_num < max(self.rho * len(self.actions) / 2, 1): #len(self.actions)-1: #1:
                return False
        for a in self.actions:
            known_num = 0
            for s in self.states:
                if self.is_known(s,a):
                    known_num += 1
            if known_num < max(self.rho * len(self.states) / 2, 1): # len(self.states)-1: #1:
                return False
        return True
    
    def complete_paras(self):
        print("doing completion----")
#        print("we have met transitions: ")
#        self.dump_transitions()
        
#        print("norm transitions")
#        self.dump_norm_trans()
        
        s_len = len(self.states)
        shape = ((len(self.actions), s_len, s_len+1))
        #para_tensor = np.zeros(shape)
        mask = np.zeros(shape)

        for s1 in self.states:
            for a in self.actions:
                for s2 in self.states:
                    #para_tensor[self.action_map[a]][self.state_map[s1]][self.state_map[s2]] = self.norm_transitions[s1][a][s2]
                    if self.is_known(s1, a):
                        mask[self.action_map[a]][self.state_map[s1]][self.state_map[s2]] = 1
#                    elif self.transitions[s1][a][s2] > self.s_a_threshold / len(self.states):
#                        mask[self.action_map[a]][self.state_map[s1]][self.state_map[s2]] = 1
                #para_tensor[self.action_map[a]][self.state_map[s1]][s_len] = self.norm_rewards[s1][a]
#                if self.r_s_a_counts[s1][a] > self.s_a_threshold / 2:
                mask[self.action_map[a]][self.state_map[s1]][s_len] = 1
        
#        print("the tensor is:")
#        print(self.par_tensor)
#        print("the mask is:")
#        print(mask)
        
#        if self.origin_tensor is not None:
#            print("the current max error: ", np.max(np.abs(para_tensor - self.origin_tensor)))
#            print("the masked max error: ", np.max(np.abs(mask*(para_tensor - self.origin_tensor))))
        
#        self.new_tensor = self.completion(shape, mask, para_tensor)
        new_tensor = np.zeros(shape)
        for i in range(len(self.states)+1):

            matrix = self.par_tensor[:,:,i]
            mask_matrix = mask[:,:,i]
#            print("the completion matrix")
#            print(matrix)
#            print(mask_matrix)
#            self.new_tensor[:,:,i] = self.completion(((len(self.actions), s_len)), mask_matrix, matrix)

#            print(matrix * mask_matrix)
#            self.new_tensor[:,:,i] = self.matrix_compl(matrix * mask_matrix, self.rank)
            
            # matrix completion that is mainly used
            if self.os:
                # opt space
                r = self.para_test(matrix)
                X, S, Y, dist = opt_space(matrix, mask_matrix, r, niter=100)
                new_tensor[:,:,i] = X @ S @ Y.T
            else:
                new_tensor[:,:,i] = self.pmf_solver(matrix, mask_matrix, self.rank, self.mu, 
                                epsilon=1e-8, max_iterations=10000)
            

#        print("the new tensor is:")
#        print(new_tensor)
        
        #combine
#        c_mask = np.ones(shape) - mask
##        print(c_mask)
#        self.new_tensor = self.new_tensor * c_mask + para_tensor
#        
#        print("the combined tensor is:")
#        print(self.new_tensor)
        
        self.par_tensor = np.copy(new_tensor)
        tt = new_tensor[:,:,:-1]
        self.par_tensor[:,:,:-1][tt < 0] = 0
        
#        
#        # rebuild the new transition prob
#        for s1 in self.states:
#            for a in self.actions:
#                for s2 in self.states:
#                    self.norm_transitions[s1][a][s2] = new_tensor[self.action_map[a]][self.state_map[s1]][self.state_map[s2]]
#                    if self.norm_transitions[s1][a][s2] < 0:
#                        self.norm_transitions[s1][a][s2] = 0 
##                if np.sum(self.new_tensor[self.action_map[a]][self.state_map[s1]]) > 0:
##                    for s2 in self.states:
##                        if mask[self.action_map[a]][self.state_map[s1]][self.state_map[s2]] == 0:
##                            self.norm_transitions[s1][a][s2] = self.new_tensor[self.action_map[a]][self.state_map[s1]][self.state_map[s2]]
##                        if self.norm_transitions[s1][a][s2] < 0:
##                            self.norm_transitions[s1][a][s2] = 0 
#                            
##                tt = self.new_tensor[self.action_map[a]][self.state_map[s1]][:-1]
##                self.new_tensor[self.action_map[a]][self.state_map[s1]][:-1][tt < 0] = 0
##                normalizer = np.sum(self.new_tensor[self.action_map[a]][self.state_map[s1]]) - self.new_tensor[self.action_map[a]][self.state_map[s1]][s_len]
###                print(self.action_map[a], self.state_map[s1], normalizer)
##                for s2 in self.state_map.keys():
##                    if normalizer is not 0:
##                        self.norm_transitions[s1][a][s2] = self.new_tensor[self.action_map[a]][self.state_map[s1]][self.state_map[s2]] / normalizer
##                    else:
##                        self.norm_transitions[s1][a][s2] = 0
#                self.norm_rewards[s1][a] = new_tensor[self.action_map[a]][self.state_map[s1]][s_len]
        
        # normalize rewards
#        ma = np.max(self.new_tensor)
#        mi = np.min(self.new_tensor)
#        
#        for s1 in self.state_map.keys():
#            for a in self.actions:
#                if self.new_tensor[self.action_map[a]][self.state_map[s1]][s_len] >= (ma - mi) / 3 * 2 + mi:
#                    self.norm_rewards[s1][a] = 1.0
#                elif self.new_tensor[self.action_map[a]][self.state_map[s1]][s_len] <= (ma - mi) / 3 + mi:
#                     self.norm_rewards[s1][a] = -1.0
#                else:
#                     self.norm_rewards[s1][a] = 0.0
                    
        print("parameters recovered")
        self.flag = True
#        self.dump_norm_trans()
#        self.par_tensor = self.get_par_tensor()
#        if self.origin_tensor is not None:
#            print("the new max error: ", np.max(np.abs(self.par_tensor - self.origin_tensor)))
        self.mask = mask
        self.update_all()
    
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
#                self.policy[s] = self.get_policy_action(s, q)
            else:
                self.policy[s] = self.get_policy_action(s, q)
#                self.policy[s] = self.get_max_q_action(s)
        print("policies updated")
        self.print_policy()
    
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
    
    def print_policy(self):
        for s in self.states:
            print("s: ", s.get_data(), "a: ",self.policy[s])
            
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

        if horizon <= 0:
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
                for s2 in self.states:
                    para_tensor[self.action_map[a]][self.state_map[s1]][self.state_map[s2]] = self.norm_transitions[s1][a][s2]
                para_tensor[self.action_map[a]][self.state_map[s1]][s_len] = self.norm_rewards[s1][a]

        return para_tensor
    
    def completion(self, shape, mask, noisy_observation, iters=10000):
        tf.reset_default_graph()
        sparsity_mask = tf.get_variable('sparsity_mask', initializer=tf.convert_to_tensor(mask, tf.float32), trainable=False)
        noisy_observation = tf.get_variable('noisy_observation', initializer=tf.convert_to_tensor(noisy_observation, tf.float32), trainable=False)
        sparse_observation = noisy_observation * sparsity_mask
        
        observed_total = tf.reduce_sum(sparsity_mask)
        initialization = t3f.random_tensor(shape, tt_rank=self.rank)
        estimated = t3f.get_variable('estimated', initializer=initialization)
        # Loss is MSE between the estimated and ground-truth tensor as computed in the observed cells.
        loss = 1.0 / observed_total * tf.reduce_sum((sparsity_mask * t3f.full(estimated) - sparse_observation)**2) 
    
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        step = optimizer.minimize(loss)
        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        last_loss = 0
        res = None
        for i in range(iters):
            _, cur_loss = sess.run([step, loss])
            
            if i % 1000 == 0:
                print(i, cur_loss)
                
            if abs(last_loss - cur_loss) < 1e-10:
                print(i, cur_loss)
                res = sess.run(t3f.full(estimated))
                print("break")
                break
            else:
                last_loss = cur_loss
        
        return res
    
    def pmf_solver(self, A, mask, k, mu, epsilon=1e-3, max_iterations=100):
        m, n = A.shape

        U = np.random.randn(m, k)
        V = np.random.randn(n, k)
    
        C_u = [np.diag(row) for row in mask]
        C_v = [np.diag(col) for col in mask.T]
    
        prev_X = np.dot(U, V.T)
    
        for _ in range(max_iterations):
    
            for i in range(m):
                U[i] = np.linalg.solve(np.linalg.multi_dot([V.T, C_u[i], V]) +
                                       mu * np.eye(k),
                                       np.linalg.multi_dot([V.T, C_u[i], A[i,:]]))
    
            for j in range(n):
                V[j] = np.linalg.solve(np.linalg.multi_dot([U.T, C_v[j], U]) +
                                       mu * np.eye(k),
                                       np.linalg.multi_dot([U.T, C_v[j], A[:,j]]))
    
            X = np.dot(U, V.T)
    
            mean_diff = np.linalg.norm(X-prev_X) / m / n
#            if _ % 100 == 0:
#                print("Iteration: %i; Mean diff: %.4f" % (_ + 1, mean_diff))
            if mean_diff < epsilon:
                break
            prev_X = X
    
        return X
    
    def matrix_compl(self, matrix, rank):
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        print("u: ", u)
        print("s: ", s)
        print("vh: ", vh)
        
        kappa = s[0] / s[rank-1]
        
        mu0 = 0
        m, n = matrix.shape
        
        for i in range(m):
            u_sum = 0
            for k in range(rank):
                u_sum += u[i][k]**2
            if mu0 < u_sum / rank:
                mu0 = u_sum / rank
                
        for j in range(n):
            v_sum = 0
            for k in range(rank):
                v_sum += vh[k][j]**2
            if mu0 < v_sum / rank:
                mu0 = v_sum / rank
    
        ss = np.copy(s)
        for k in range(rank):
            ss[k] = s[k] / s[0]
        
        mu1 = np.trace(np.dot(u[:,:rank]*ss[:rank],vh[:rank,:])) / math.sqrt(rank)
        
        print("mu0=", mu0)
        print("mu1=", mu1)
        print("kappa=", kappa)
        print("trimming: ")
        trim = np.dot(u[:,:rank]*s[:rank],vh[:rank,:])
        
        print(trim)
    
        return trim
    
    def para_test(self, matrix, rank=None):
        m, n = np.shape(matrix)
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    #        print("u: ", u)
        print("s: ", s)
    #        print("vh: ", vh)
        if rank is None:
            for r in range(min(m, n)):
                if abs(s[r]) < 1e-8 or abs(s[0]/s[r]) > 20:
                    rank = r
                    break
            if rank is None:
                rank = min(m, n)
        print("rank=", rank)
        
    #    u = math.sqrt(m) * u
    #    vh = math.sqrt(n) * vh
        
        kappa = s[0] / s[rank-1]
        
        mu0 = 0
        for i in range(m):
            u_sum = 0
            for k in range(rank):
                u_sum += u[i][k]**2
            if mu0 < u_sum / rank * m:
                mu0 = u_sum / rank * m
                
        for j in range(n):
            v_sum = 0
            for k in range(rank):
                v_sum += vh[k][j]**2
            if mu0 < v_sum / rank * n:
                mu0 = v_sum / rank * n
    
        ss = np.copy(s)
        for k in range(rank):
            ss[k] = s[k] / s[0]
            
        
        max_entry = np.max(np.abs(np.outer(u[:,:rank], vh[:rank,:])))
        mu1 = max_entry * math.sqrt(m * n / rank)
        
    #    mu1 = abs(np.trace(np.dot(u[:,:rank]*ss[:rank],vh[:rank,:]))) / math.sqrt(rank)
        
        print("mu0=", mu0)
        print("mu1=", mu1)
        print("kappa=", kappa)
        
    #    proj = np.dot(u[:,:rank]*s[:rank],vh[:rank,:])
    #    print("r-projection: ")
    #    print(proj)
        
        return rank