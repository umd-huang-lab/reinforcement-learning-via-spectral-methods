
try:
    xrange
except NameError:
    xrange = range

# Fix input to cooperate with python 2 and 3.
try:
   input = raw_input
except NameError:
   pass

# Imports.
import tensor_rl.agents, tensor_rl.experiments, tensor_rl.mdp, tensor_rl.tasks, tensor_rl.utils
import tensor_rl.run_experiments
