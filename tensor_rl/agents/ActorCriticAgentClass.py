''' ActorCriticAgentClass.py: Class for a REINFORCE agent, from:

    Williams, Ronald J. "Simple statistical gradient-following algorithms for
    connectionist reinforcement learning." Machine learning 8.3-4 (1992): 229-256.
'''

# Python imports.
from collections import defaultdict
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K

import tensorflow as tf

import random
import numpy as np
from collections import deque

# Other imports
from tensor_rl.agents.AgentClass import Agent

class ActorCriticAgent(Agent):
    ''' Class for a random decision maker. '''

    def __init__(self, actions, n_states, sess, name="actor-critic"):
        name = "policy_gradient" if name is "" else name
        Agent.__init__(self, name=name, actions=actions)
        
        self.reset()
        self.sess = sess
        self.n_states = n_states
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.gamma = .95
        self.tau   = .125
        
        self.memory = deque(maxlen=2000)
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        self.actor_critic_grad = tf.placeholder(tf.float32, [None, len(self.actions)]) # where we will feed de/dC (from critic)
        
        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        self.critic_state_input, self.critic_action_input, \
        	self.critic_model = self.create_critic_model()
            
        _, _, self.target_critic_model = self.create_critic_model()

        self.critic_grads = tf.gradients(self.critic_model.output, 
        	self.critic_action_input) # where we calcaulte de/dC for feeding above
        
        # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())
        
    def reset(self):
        '''
        Summary:
            Resets the agent back to its tabula rasa config.
        '''
        self.prev_state = None
        self.prev_action = None
        self.action_map = {}
        k = 0
        for a in self.actions:
            self.action_map[a] = k
            k += 1
        print(self.action_map)
    
    def encode_state(self, state):
        state_code = np.zeros(shape=(1, self.n_states))
        state_code[0][state.get_data()] = 1
        return state_code

    def create_actor_model(self):
        state_input = Input(shape=(self.n_states,))
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        output = Dense(len(self.actions), activation='relu')(h3)
        
        model = Model(input=state_input, output=output)
        adam  = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape=(self.n_states,))
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48)(state_h1)
        
        action_input = Input(shape=(len(self.actions),))
        action_h1    = Dense(48)(action_input)
        
        merged    = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model  = Model(input=[state_input,action_input], output=output)
        
        adam  = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model


    def _train_actor(self, samples):
        for sample in samples:
#            print("train actor")
            cur_state, action, reward, new_state, _ = sample
            if cur_state is not None:
                predicted_action = self.actor_model.predict(cur_state)
                grads = self.sess.run(self.critic_grads, feed_dict={
                        self.critic_state_input:  cur_state,
                        self.critic_action_input: predicted_action
                	})[0]
        
                self.sess.run(self.optimize, feed_dict={
                        self.actor_state_input: cur_state,
                        self.actor_critic_grad: grads
                	})

    def _train_critic(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            if cur_state is not None:
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.target_critic_model.predict([new_state, target_action])[0][0]
                reward += self.gamma * future_reward
#                print("fit", [cur_state, action])
                self.critic_model.fit([cur_state.astype('float32'), action.astype('float32')], [reward], verbose=0)
#                print("done fit")
        
    def train(self):
        batch_size = 32
        if len(self.memory) < batch_size:
        	return

        rewards = []
        samples = random.sample(self.memory, batch_size)
        self._train_critic(samples)
        self._train_actor(samples)

	# ========================================================================= #
	#                         Target Model Updating                             #
	# ========================================================================= #

    def _update_actor_target(self):
        actor_model_weights  = self.actor_model.get_weights()
        actor_target_weights = self.target_critic_model.get_weights()
        
        for i in range(len(actor_target_weights)):
        	actor_target_weights[i] = actor_model_weights[i]
        self.target_critic_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights  = self.critic_model.get_weights()
        critic_target_weights = self.critic_target_model.get_weights()
        
        for i in range(len(critic_target_weights)):
        	critic_target_weights[i] = critic_model_weights[i]
        self.critic_target_model.set_weights(critic_target_weights)        

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()
        
    def act(self, state, reward):
        '''
        Args:
            state (State)
            reward (float)

        Returns:
            (str)
        '''
        state = self.encode_state(state)
        self.update(self.prev_state, self.prev_action, reward, state)

        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            action = random.choice(self.actions)
            predict = np.zeros(shape=(1,len(self.actions)))
            predict[0][action] = 1
            # predict[0][self.action_map[action]] = 1
        else:
            predict = self.actor_model.predict(state)
            action = np.argmax(predict)
#            print("pred", end=" ")
        # print("action:", action)
        
        self.prev_action = predict
        self.prev_state = state
        
        return action

    def update(self, state, action, reward, next_state):
        '''
        Args:
            state (State)
            action (str)
            reward (float)
            next_state (State)

        Summary:
            Perform a state of policy gradient.
        '''
        self.memory.append([state, action, reward, next_state, False])
        self.train()
