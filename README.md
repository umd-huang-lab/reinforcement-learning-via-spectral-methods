# reinforcement-learning-via-spectral-methods
Model-based reinforcement learning algorithms make decisions by building and utilizing a model of the environment. However, none of the existing algorithms attempts to infer the dynamics of any state-action pair from known state-action pairs before meeting it for sufficient times. We propose a new model-based method called Greedy Inference Model (GIM) that infers the unknown dynamics from known dynamics based on the internal spectral properties of the environment. In other words, GIM can “learn by analogy”. We further introduce a new exploration strategy which ensures that the agent rapidly and evenly visits unknown state-action pairs. GIM is much more computationally efficient than state-of-the-art model-based algorithms, as the number of dynamic programming operations is independent of the environment size. Lower sample complexity could also be achieved under mild conditions compared against methods without inferring. Experimental results demon- strate the effectiveness and efficiency of GIM in a variety of real-world tasks.

>  Paper Link: <https://arxiv.org/abs/1912.10329>

---

Our implementation was modified from [simple_rl](<https://github.com/david-abel/simple_rl/tree/master/simple_rl>). 

See *examples/simple_example.py* for how to use GIM. Simply uncomment  line 151~154 for multiple tasks that are shown in the paper. The sample experiment results are shown under the folder *examples/sample_results*.