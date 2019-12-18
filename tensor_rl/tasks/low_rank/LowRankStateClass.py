#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:04:55 2019

"""

from tensor_rl.mdp.StateClass import State

class LowRankState(State):
    ''' Class for Low Rank MDP States '''

    def __init__(self, x):
        State.__init__(self, data=x)
        self.x = int(x)
    
    def get_num(self):
        return self.x

    def __hash__(self):
        return hash(self.data)

    def __str__(self):
        return "s: (" + str(self.x) + ")"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, LowRankState) and self.x == other.x 