''' GridWorldMDPClass.py: Contains the GridWorldMDP class. '''

# Python imports.
from __future__ import print_function
import random
import sys, os, math
import numpy as np
from collections import defaultdict

# Other imports.
from tensor_rl.mdp.MDPClass import MDP
from tensor_rl.tasks.grid_world.GridWorldStateClass import GridWorldState

# Fix input to cooperate with python 2 and 3.
try:
   input = raw_input
except NameError:
   pass

class GridWorldMDP(MDP):
    ''' Class for a Grid World MDP '''

    # Static constants.
    ACTIONS = ["up", "down", "left", "right"]

    def __init__(self,
                width=5,
                height=3,
                init_loc=(1, 1),
                rand_init=False,
                goal_locs=[(5, 3)],
                lava_locs=[()],
                walls=[],
                is_goal_terminal=True,
                gamma=0.99,
                slip_prob=0.0,
                step_cost=0.0,
                lava_cost=1.0,
                name="gridworld"):
        '''
        Args:
            height (int)
            width (int)
            init_loc (tuple: (int, int))
            goal_locs (list of tuples: [(int, int)...])
            lava_locs (list of tuples: [(int, int)...]): These locations return -1 reward.
            walls (list)
            is_goal_terminal (bool)
        '''

        # Setup init location.
        self.rand_init = rand_init
        if rand_init:
            init_loc = random.randint(1, width), random.randint(1, height)
            while init_loc in walls:
                init_loc = random.randint(1, width), random.randint(1, height)
        self.init_loc = init_loc
        init_state = GridWorldState(init_loc[0], init_loc[1])

        MDP.__init__(self, GridWorldMDP.ACTIONS, self._transition_func, self._reward_func, init_state=init_state, gamma=gamma)

        if type(goal_locs) is not list:
            raise ValueError("(tensor_rl) GridWorld Error: argument @goal_locs needs to be a list of locations. For example: [(3,3), (4,3)].")
        self.step_cost = step_cost
        self.lava_cost = lava_cost
        self.walls = walls
        self.width = width
        self.height = height
        self.goal_locs = goal_locs
        
        self._init_states()
        self.cur_state = GridWorldState(init_loc[0], init_loc[1])
        
        self.is_goal_terminal = is_goal_terminal
        self.slip_prob = slip_prob
        self.name = name
        self.lava_locs = lava_locs

    def _init_states(self):
        self.states = []
        self.state_map = {}
        for i in range(self.height):
            for j in range(self.width):
                self.states.append(GridWorldState(j+1, i+1))
                self.state_map[GridWorldState(j+1, i+1)] = j + i * self.width
                
    def get_par_tensor(self):
        n_states = self.height * self.width
        par_tensor = np.zeros((len(GridWorldMDP.ACTIONS), n_states, n_states+1))
        for a in range(len(GridWorldMDP.ACTIONS)):
            action = GridWorldMDP.ACTIONS[a]
            matrix = self.gen_map(self.height, self.width, action)
            par_tensor[a,:,:-1] = matrix
            
            for s1 in range(n_states):
                re = 0
                for s2 in range(n_states):
                    re += matrix[s1][s2] * self.get_state_reward(s2)
                par_tensor[a,s1,-1] = re
        return par_tensor
    
    def get_state_reward(self, s):
        y = int(math.ceil((s + 1) / self.width))
        x = int((s + 1) - self.width * (y - 1))
#        print(s, x, y)
        if (x, y) in self.goal_locs:
            return 1.0
        elif (x, y) in self.lava_locs:
            return - self.lava_cost
        else:
            return 0 - self.step_cost
    
    def gen_map(self, h, w, action):
        prob_right = 1 - self.slip_prob
        prob_wrong = self.slip_prob / 2
        M = np.zeros((h*w, h*w))
        if action == "right":
            for i in range(h):
                for j in range(w):
                    idx = i*w + j
                    if j < w-1:
                        M[idx][idx+1] += prob_right
                    else:
                        M[idx][idx] += prob_right
                    if i > 0:
                        M[idx][idx-w] += prob_wrong
                    else:
                        M[idx][idx] += prob_wrong
                    if i < h-1:
                        M[idx][idx+w] += prob_wrong
                    else:
                        M[idx][idx] += prob_wrong
        elif action == "left":
            for i in range(h):
                for j in range(w):
                    idx = i*w + j
                    if j > 0:
                        M[idx][idx-1] += prob_right
                    else:
                        M[idx][idx] += prob_right
                    if i > 0:
                        M[idx][idx-w] += prob_wrong
                    else:
                        M[idx][idx] += prob_wrong
                    if i < h-1:
                        M[idx][idx+w] += prob_wrong
                    else:
                        M[idx][idx] += prob_wrong
        elif action == "up":
            for i in range(h):
                for j in range(w):
                    idx = i*w + j
                    if i < h-1:
                        M[idx][idx+w] += prob_right
                    else:
                        M[idx][idx] += prob_right
                    if j > 0:
                        M[idx][idx-1] += prob_wrong
                    else:
                        M[idx][idx] += prob_wrong
                    if j < w-1:
                        M[idx][idx+1] += prob_wrong
                    else:
                        M[idx][idx] += prob_wrong
        elif action == "down":
            for i in range(h):
                for j in range(w):
                    idx = i*w + j
                    if i > 0:
                        M[idx][idx-w] += prob_right
                    else:
                        M[idx][idx] += prob_right
                    if j > 0:
                        M[idx][idx-1] += prob_wrong
                    else:
                        M[idx][idx] += prob_wrong
                    if j < w-1:
                        M[idx][idx+1] += prob_wrong
                    else:
                        M[idx][idx] += prob_wrong
        return M
    
    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        param_dict = defaultdict(int)
        param_dict["width"] = self.width
        param_dict["height"] = self.height
        param_dict["init_loc"] = self.init_loc
        param_dict["rand_init"] = self.rand_init
        param_dict["goal_locs"] = self.goal_locs
        param_dict["lava_locs"] = self.lava_locs
        param_dict["walls"] = self.walls
        param_dict["is_goal_terminal"] = self.is_goal_terminal
        param_dict["gamma"] = self.gamma
        param_dict["slip_prob"] = self.slip_prob
        param_dict["step_cost"] = self.step_cost
        param_dict["lava_cost"] = self.lava_cost
   
        return param_dict

    def set_slip_prob(self, slip_prob):
        self.slip_prob = slip_prob

    def get_slip_prob(self):
        return self.slip_prob

    def is_goal_state(self, state):
        return (state.x, state.y) in self.goal_locs

    def _reward_func(self, state, action, next_state):
        '''
        Args:
            state (State)
            action (str)
            next_state (State)

        Returns
            (float)
        '''
        if (int(next_state.x), int(next_state.y)) in self.goal_locs:
            # self._is_goal_state_action(state, action):
            return 1.0 - self.step_cost
        elif (int(next_state.x), int(next_state.y)) in self.lava_locs:
            return - self.lava_cost
        else:
            return 0 - self.step_cost

    def _is_goal_state_action(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns:
            (bool): True iff the state-action pair send the agent to the goal state.
        '''
        if (state.x, state.y) in self.goal_locs and self.is_goal_terminal:
            # Already at terminal.
            return False

        if action == "left" and (state.x - 1, state.y) in self.goal_locs:
            return True
        elif action == "right" and (state.x + 1, state.y) in self.goal_locs:
            return True
        elif action == "down" and (state.x, state.y - 1) in self.goal_locs:
            return True
        elif action == "up" and (state.x, state.y + 1) in self.goal_locs:
            return True
        else:
            return False

    def _transition_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (State)
        '''
#        if state.is_terminal():
#            return state
#        print("cur", state, action)
#        define my own slip rules to make it low rank
#        if not(self._is_goal_state_action(state, action)) and self.slip_prob > random.random():
        # I will never stop
        if self.slip_prob > random.random():
            if action == "up":
                if random.random() > 0.5:
                    action = "left"
                else:
                    action = "right"
            elif action == "down":
                if random.random() > 0.5:
                    action = "left"
                else:
                    action = "right"
            elif action == "left":
                if random.random() > 0.5:
                    action = "up"
                else:
                    action = "down"
            elif action == "right":
                if random.random() > 0.5:
                    action = "up"
                else:
                    action = "down"
        
        
#        if not(self._is_goal_state_action(state, action)) and self.slip_prob > random.random():
#            # Flip dir.
#            if action == "up":
#                action = random.choice(["left", "right"])
#            elif action == "down":
#                action = random.choice(["left", "right"])
#            elif action == "left":
#                action = random.choice(["up", "down"])
#            elif action == "right":
#                action = random.choice(["up", "down"])

        if action == "up" and state.y < self.height and not self.is_wall(state.x, state.y + 1):
            next_state = GridWorldState(state.x, state.y + 1)
        elif action == "down" and state.y > 1 and not self.is_wall(state.x, state.y - 1):
            next_state = GridWorldState(state.x, state.y - 1)
        elif action == "right" and state.x < self.width and not self.is_wall(state.x + 1, state.y):
            next_state = GridWorldState(state.x + 1, state.y)
        elif action == "left" and state.x > 1 and not self.is_wall(state.x - 1, state.y):
            next_state = GridWorldState(state.x - 1, state.y)
        else:
            next_state = GridWorldState(state.x, state.y)

#        if (next_state.x, next_state.y) in self.goal_locs and self.is_goal_terminal:
#            next_state.set_terminal(True)
        
#        print(action, next_state)
        return next_state

    def is_wall(self, x, y):
        '''
        Args:
            x (int)
            y (int)

        Returns:
            (bool): True iff (x,y) is a wall location.
        '''

        return (x, y) in self.walls

    def __str__(self):
        return self.name + "_h-" + str(self.height) + "_w-" + str(self.width)

    def __repr__(self):
        return self.__str__()

    def get_goal_locs(self):
        return self.goal_locs

    def get_lava_locs(self):
        return self.lava_locs

    def visualize_policy(self, policy):
        from tensor_rl.utils import mdp_visualizer as mdpv
        from tensor_rl.tasks.grid_world.grid_visualizer import _draw_state

        action_char_dict = {
            "up":"^",       #u"\u2191",
            "down":"v",     #u"\u2193",
            "left":"<",     #u"\u2190",
            "right":">",    #u"\u2192"
        }

        mdpv.visualize_policy(self, policy, _draw_state, action_char_dict)

    def visualize_agent(self, agent):
        from tensor_rl.utils import mdp_visualizer as mdpv
        from tensor_rl.tasks.grid_world.grid_visualizer import _draw_state
        mdpv.visualize_agent(self, agent, _draw_state)

    def visualize_value(self):
        from tensor_rl.utils import mdp_visualizer as mdpv
        from tensor_rl.tasks.grid_world.grid_visualizer import _draw_state
        mdpv.visualize_value(self, _draw_state)

    def visualize_learning(self, agent, delay=0.0):
        from tensor_rl.utils import mdp_visualizer as mdpv
        from tensor_rl.tasks.grid_world.grid_visualizer import _draw_state
        mdpv.visualize_learning(self, agent, _draw_state, delay=delay)

    def visualize_interaction(self):
        from tensor_rl.utils import mdp_visualizer as mdpv
        from tensor_rl.tasks.grid_world.grid_visualizer import _draw_state
        mdpv.visualize_interaction(self, _draw_state)

def _error_check(state, action):
    '''
    Args:
        state (State)
        action (str)

    Summary:
        Checks to make sure the received state and action are of the right type.
    '''

    if action not in GridWorldMDP.ACTIONS:
        raise ValueError("(tensor_rl) GridWorldError: the action provided (" + str(action) + ") was invalid in state: " + str(state) + ".")

    if not isinstance(state, GridWorldState):
        raise ValueError("(tensor_rl) GridWorldError: the given state (" + str(state) + ") was not of the correct class.")

def make_grid_world_from_file(file_name, randomize=False, num_goals=1, name=None, goal_num=None, slip_prob=0.0):
    '''
    Args:
        file_name (str)
        randomize (bool): If true, chooses a random agent location and goal location.
        num_goals (int)
        name (str)

    Returns:
        (GridWorldMDP)

    Summary:
        Builds a GridWorldMDP from a file:
            'w' --> wall
            'a' --> agent
            'g' --> goal
            '-' --> empty
    '''

    if name is None:
        name = file_name.split(".")[0]

    # grid_path = os.path.dirname(os.path.realpath(__file__))
    wall_file = open(os.path.join(os.getcwd(), file_name))
    wall_lines = wall_file.readlines()

    # Get walls, agent, goal loc.
    num_rows = len(wall_lines)
    num_cols = len(wall_lines[0].strip())
    empty_cells = []
    agent_x, agent_y = 1, 1
    walls = []
    goal_locs = []
    lava_locs = []

    for i, line in enumerate(wall_lines):
        line = line.strip()
        for j, ch in enumerate(line):
            if ch == "w":
                walls.append((j + 1, num_rows - i))
            elif ch == "g":
                goal_locs.append((j + 1, num_rows - i))
            elif ch == "l":
                lava_locs.append((j + 1, num_rows - i))
            elif ch == "a":
                agent_x, agent_y = j + 1, num_rows - i
            elif ch == "-":
                empty_cells.append((j + 1, num_rows - i))

    if goal_num is not None:
        goal_locs = [goal_locs[goal_num % len(goal_locs)]]

    if randomize:
        agent_x, agent_y = random.choice(empty_cells)
        if len(goal_locs) == 0:
            # Sample @num_goals random goal locations.
            goal_locs = random.sample(empty_cells, num_goals)
        else:
            goal_locs = random.sample(goal_locs, num_goals)

    if len(goal_locs) == 0:
        goal_locs = [(num_cols, num_rows)]

    return GridWorldMDP(width=num_cols, height=num_rows, init_loc=(agent_x, agent_y), goal_locs=goal_locs, lava_locs=lava_locs, walls=walls, name=name, slip_prob=slip_prob)

    def reset(self):
        if self.rand_init:
            init_loc = random.randint(1, width), random.randint(1, height)
            self.cur_state = GridWorldState(init_loc[0], init_loc[1])
        else:
            self.cur_state = copy.deepcopy(self.init_state)

def main():
    grid_world = GridWorldMDP(5, 10, (1, 1), (6, 7))

    grid_world.visualize()

if __name__ == "__main__":
    main()
