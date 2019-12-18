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
from tensor_rl.tasks.low_rank.LowRankStateClass import LowRankState

# Fix input to cooperate with python 2 and 3.
try:
   input = raw_input
except NameError:
   pass

class LowRankMDP(MDP):
    ''' Class for a Low Rank MDP '''

    # Static constants.
#    ACTIONS = [0, 1, 2]
#    N_ACTIONS = 3

    def __init__(self,
                 n_states,
                 n_actions,
#                 goal_states,
                 penal_states = None,
                 rank = 1,
                 mu_max = 2,
                 kappa_max = 2,
                 factors = None,
                 init_state = 0,
                 gamma = 0.5,
                 name="lowrank",
                 par_tensor=None):
        self.n_actions = n_actions
        self.actions = list(range(n_actions))
        self.n_states = n_states
        self.rank = rank
        self.mu_max = mu_max
        self.kappa_max = kappa_max
        # generate transition matrices
        if par_tensor is not None:
            self.par_tensor = par_tensor
        else:
            self.par_tensor = np.zeros((n_actions, n_states, n_states+1))
            for s in range(n_states+1):
                B = np.round_(np.random.rand(rank, n_states), 1)
                A = np.round_(np.random.rand(n_actions, rank), 1) #np.random.randint(1, n_states, size=(LowRankMDP.N_ACTIONS, rank))
                matrix = A @ B
    #            print(matrix)
                # restrict reward
                if s == n_states:
                    ma = np.max(matrix)
                    mi = np.min(matrix)
                    for i in range(n_actions):
                        for j in range(n_states):
                            matrix[i][j] = (matrix[i][j]-mi) / (ma-mi)
    #                        if matrix[i][j] >= (ma - mi) / 2 + mi:
    #                            matrix[i][j] = 1.0
    #                        elif matrix[i][j] <= (ma - mi) / 6 + mi:
    #                            matrix[i][j] = -1.0
    #                        else:
    #                            matrix[i][j] = 0.0
                self.par_tensor[:,:,s] = matrix
                
            
    #        self.par_tensor = None
    #        for k in range(n_actions):
    #            B = np.round_(np.random.rand(rank, n_states+1), 1)
    #            A = np.random.randint(1, n_states, size=(n_states, rank))
    #            matrix = A @ B
    #            if self.par_tensor is None:
    #                self.par_tensor = [matrix]
    #            else:
    #                self.par_tensor = np.concatenate((self.par_tensor, [matrix]), axis=0)
    #        print(self.par_tensor)
            
            # normalize rewards
    #        ma = np.max(self.par_tensor)
    #        mi = np.min(self.par_tensor)
            
    #        for i in range(LowRankMDP.N_ACTIONS):
    #            for j in range(n_states):
    #                if self.par_tensor[i][j][n_states] >= (ma - mi) / 3 * 2 + mi:
    #                    self.par_tensor[i][j][n_states] = 1.0
    #                elif self.par_tensor[i][j][n_states] <= (ma - mi) / 3 + mi:
    #                    self.par_tensor[i][j][n_states] = -1.0
    #                else:
    #                    self.par_tensor[i][j][n_states] = 0.0
            
            if self.rank > 1:
                self.par_tensor = self._init_dynamics()
    #        self.par_tensor = self._init_dynamics2()
            # normalization
            for i in range(n_actions):
                for j in range(n_states):
                    if np.sum(self.par_tensor[i][j][:n_states]) != 0:
                        self.par_tensor[i][j][:n_states] = self.par_tensor[i][j][:n_states] * 1.0 / np.sum(self.par_tensor[i][j][:n_states])  
            
    #        if factors is None:
    #            factors = np.random.rand(2, rank, n_states)
    #        
    #        a_factors = np.random.rand(rank, LowRankMDP.N_ACTIONS)
    #        
    #        print("a", a_factors)
    #        
    #        comps = []
    #        for i in range(rank):
    #            comps.append(np.multiply.outer(a_factors[i], np.multiply.outer(factors[0][i], factors[1][i])))
    #        
    #        self.par_tensor = np.sum(comps, axis=0)
    #        print(self.par_tensor)
    #        # doing normalization
    #        for i in range(LowRankMDP.N_ACTIONS):
    #            for j in range(n_states):
    #                if np.sum(self.par_tensor[i][j]) != 0:
    #                    self.par_tensor[i][j] = self.par_tensor[i][j] * 1.0 / np.sum(self.par_tensor[i][j])
    #        print("the transition tensor is:")
    #        print(self.par_tensor)
    #        self.par_tensor = np.array([[[0, .5, 0, .5, 0],
    #                  [0, .3, 0, .3, .4],
    #                  [0, 0, 0, 0, 1],
    #                  [0, 0, .2, 0, .8],
    #                  [.1, .1, .3, .3, .2]],
    #    
    #                  [[0, 0, 0, 0, 1],
    #                  [0, .5, 0, .5, 0],
    #                  [0, .3, 0, .3, .4],
    #                  [0, 0, .2, 0, .8],
    #                  [.1, .1, .3, .3, .2]],
    #                   
    #                  [[0, 0, 0, 0, 1],
    #                  [0, .5, 0, .5, 0],
    #                  [0, .3, 0, .3, .4],
    #                  [0, 0, .2, 0, .8],
    #                  [.1, .1, .3, .3, .2]]])
    #        self.par_tensor = np.array([[[.1, .2, .3, .2, .2, 1],
    #                  [.1, .2, .3, .2, .2, 1],
    #                  [.1, .2, .3, .2, .2, 1],
    #                  [.1, .2, .3, .2, .2, 1],
    #                  [.1, .2, .3, .2, .2, 1]],
    #    
    #                  [[.2, .1, .3, .3, .1, -1],
    #                  [.2, .1, .3, .3, .1, -1],
    #                  [.2, .1, .3, .3, .1, -1],
    #                  [.2, .1, .3, .3, .1, -1],
    #                  [.2, .1, .3, .3, .1, -1]],
    #                   
    #                  [[.2, .1, .3, .3, .1, 0],
    #                  [.2, .1, .3, .3, .1, 0],
    #                  [.2, .1, .3, .3, .1, 0],
    #                  [.2, .1, .3, .3, .1, 0],
    #                  [.2, .1, .3, .3, .1, 0]]])
    #        self.par_tensor = self._init_dynamics()
            print("the transition tensor is:")
            print(self.par_tensor)
        
            # test parameters
            for s in range(n_states+1):
                print("the ", s, "-th matrix")
                projection = self.para_test(self.par_tensor[:,:,s], n_actions, n_states)
                print("projection error: ", np.sum((self.par_tensor[:,:,s] - projection)**2))

        # init state
        self._init_states()
        init_state = self.states[init_state]
        print("states ", self.states)
        print("state map ", self.state_map)
        
        print("init:", init_state)
#        self.goal_states = goal_states
#        self.penal_states = penal_states
        
        MDP.__init__(self, self.actions, self._transition_func, self._reward_func, init_state=init_state, gamma=gamma)
        
        self.name = name
    
    def para_test(self, matrix, m, n, rank=None):
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    #        print("u: ", u)
#        print("s: ", s)
    #        print("vh: ", vh)
        if rank is None:
            for r in range(1, min(m, n)):
                print(abs(s[0]/s[r]))
                if abs(s[r]) < 1e-8 or abs(s[0]/s[r]) > 10:
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
#        print("r-projection: ")
        proj = np.dot(u[:,:rank]*s[:rank],vh[:rank,:])
#        print(proj)
        
        return proj
        
    
    def _init_dynamics(self):
        n = self.n_states
        tensor = np.zeros((self.n_actions, n, n + 1))
#        print(tensor)
#        for j in range(LowRankMDP.N_ACTIONS):
#            for k in range(self.n_states):
#                tensor[j][k][np.where(tensor[j,k,:n]<0)] = 0
#                if np.sum(tensor[j,k,:n]) > 0:
#                    tensor[j,k,:n] = tensor[j,k,:n] / np.sum(tensor[j,k,:n])
#                elif np.sum(tensor[j,k,:n]) == 0:
#                    print(np.sum(tensor[j,k,:n]))
#                    tensor[j][k][0] = 1.0
#        return tensor
    
        for s in range(n+1):
            matrix = self.generate_good(self.n_actions, n, self.rank, self.mu_max, self.kappa_max) #np.random.rand(LowRankMDP.N_ACTIONS, n)
            if s < n:
                matrix[np.where(matrix<0)] = 0
            
#            elif s == n:
#                ma = np.max(matrix)
#                mi = np.min(matrix)
#                for i in range(self.n_actions):
#                    for j in range(n):
#                        matrix[i][j] = (matrix[i][j]-mi) / (ma-mi)
            tensor[:,:,s] = matrix
        
#        tensor[np.where(tensor<0)] = 0
        return tensor
    
    def _init_dynamics2(self):
        n = self.n_states
        tensor = np.zeros((self.n_actions, n, n + 1))
    
        for s in range(n+1):
            matrix = self.generate_good(self.n_actions, n, self.rank, self.n_actions / self.rank) #np.random.rand(LowRankMDP.N_ACTIONS, n)
            if s < n:
                matrix[np.where(matrix<0)] = 0
            zero_rate = 0.7
            for i in range(self.n_actions):
                for j in range(n):
                    if random.random() <= zero_rate:
                        matrix[i][j] = 0
            tensor[:,:,s] = matrix
        
        return tensor
    
    def generate_good(self, m, n, rank, mu=2, ka=2):
        """To make the mu0, mu1 and the kappa smaller"""
        sr = random.random()
        s = []
        s.append(sr)
        for r in range(rank-1):
            newele = s[-1] * (1 + ka * random.random() / (rank-1))
            s.append(newele)
        s.reverse()
        
        # best_u = None
        # best_mu0 = 0
        # while best_mu0 == 0:
        #     for _ in range(10):
        #         A = np.random.rand(m,m)
        #         A = scipy.linalg.orth(A)
        #         u = A[:, :rank]
        #         mu0 = self.compute_mu(u, m, rank)
        #         print("mu0 : ", mu0)
        #         if mu0 <= mu and mu0 >= best_mu0:
        #             best_mu0 = mu0
        #             best_u = u
        # print("mu0 for u:", best_mu0)
        # # print(u.T @ u)
        
        # best_v = None
        # best_mu0 = 0
        # while best_mu0 == 0:
        #     for _ in range(10):
        #         B = np.random.rand(n,n)
        #         B = scipy.linalg.orth(B)
        #         v = B[:, :rank]
        #         mu0 = self.compute_mu(v, n, rank)
        #         print("mu0 : ", mu0)
        #         if mu0 <= mu and mu0 >= best_mu0:
        #             best_mu0 = mu0
        #             best_v = v
        # print("mu0 for v:", best_mu0)
        # u = best_u
        # v = best_v

        for _ in range(100):
            A = np.random.rand(m,m)
            A = scipy.linalg.orth(A)
            u = A[:, :rank]
            mu0 = self.compute_mu(u, m, rank)
            print("mu0 : ", mu0)
            if mu0 <= mu:
                break
        print("mu0 for u:", mu0)    

        for _ in range(10):
            B = np.random.rand(n,n)
            B = scipy.linalg.orth(B)
            v = B[:, :rank]
            mu0 = self.compute_mu(v, n, rank)
            print("mu0 : ", mu0)
            if mu0 <= mu:
                break
        print("mu0 for both:", mu0)

        matrix = np.dot(u*s, v.T)
        
        kappa = s[0] / s[-1]
        print("kappa=", kappa)
        
        ss = np.copy(s)
        for k in range(rank):
            ss[k] = s[k] / s[0]
        
        max_entry = np.max(np.abs(np.outer(u[:,:rank], v.T[:rank,:])))
        mu1 = max_entry * math.sqrt(m * n / rank)
        print("mu1=", mu1)
        
        return matrix
    
    def compute_mu(self, u, m, r):
        mu0 = 0
        for i in range(m):
            u_sum = 0
            for k in range(r):
                u_sum += u[i][k]**2
            if mu0 < u_sum / r * m:
                mu0 = u_sum / r * m
        return mu0
    
    def _init_states(self):
        self.states = []
        self.state_map = {}
        for i in range(self.n_states):
            self.states.append(LowRankState(i))
            self.state_map[LowRankState(i)] = i
    
    def _choose_by_dist(self, prob_vec):
        ''' return a chosen state number '''
        
        rand = random.random()
#        print("a random number:", rand)
#        print("prob_dist", prob_vec)
        s = 0
        for i in range(len(prob_vec)-1):
            s += prob_vec[i]
            if s > rand:
                return LowRankState(i)
        print("nothing has been chosen!")
        return None
            
    
    def _transition_func(self, state, action):
#        print("(", action, ")", end=" ")
        next_state = self._choose_by_dist(self.par_tensor[action][state.get_data()])
#        print("next", next_state, end=' ')
        return next_state
        
        
    def _reward_func(self, state, action, next_state):
#        if self.par_tensor[action][state.get_data()][self.n_states] >= 1:
#            return 1.0
#        else:
#            return 0
        return self.par_tensor[action][state.get_data()][self.n_states]
#        if next_state.get_data() in self.goal_states:
#            return 1.0
#        elif next_state.get_data() == self.penal_states:
#            return - 1.0
#        else:
#            return 0.0
    
    def __str__(self):
        return self.name + "_S=" + str(self.n_states) + "_A=" + str(self.n_actions) + "_r=" + str(self.rank)
    
if __name__ == "__main__":
    # 3 states
    mdp = LowRankMDP(6, [2], rank=5)
    
    
    
    
    

    