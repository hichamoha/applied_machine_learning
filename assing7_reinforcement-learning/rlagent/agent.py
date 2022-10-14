"""agent.py: Contains the entire deep reinforcement learning agent."""
__author__ = "Erik Gärtner"

from collections import deque

import tensorflow as tf
import numpy as np

from .expreplay import ExpReplay


class Agent():
    """
    The agent class where you should implement the vanilla policy gradient agent.
    """

    def __init__(self, tf_session, state_size=(4,), action_size=2,
                 learning_rate=1e-3, gamma=0.99, memory_size=5000):
        """
        The initialization function. Besides saving attributes we also need
        to create the policy network in Tensorflow that later will be used.
        """

        self.state_size = state_size
        self.action_size = action_size
        self.tf_sess = tf_session
        self.gamma = gamma
        self.replay = ExpReplay(memory_size)

        with tf.variable_scope('agent'):
            # Create tf placeholders, i.e. inputs into the network graph.

            # Create the hidden layers

            # Create the loss. We need to multiply the reward with the
            # log-probability of the selected actions.

            # Create the optimizer to minimize the loss

            pass

        tf_session.run(tf.global_variables_initializer())

    def take_action(self, state):
        """
        Given the current state sample an action from the policy network.
        Return a the index of the action [0..N).
        """
        action = np.random.choice(range(self.action_size))
        return action

    def record_action(self, state0, action, reward, state1, done):
        """
        Record an action taken by the action and the associated reward
        and next state. This will later be used for traning.
        """
        pass

    def train_agent(self):
        """
        Train the policy network using the collected experiences during the
        episode(s).
        """
        # Retrieve collected experiences from memory

        # Discount and normalize rewards

        # Shuffle for better learning

        # Feed the experiences through the network with rewards to compute and
        # minimize the loss.
        pass

    def discount_rewards_and_normalize(self, rewards):
        """
        Given the rewards for an epsiode discount them by gamma.
        Next since we are sending them into the neural network they should
        have a zero mean and unit variance.

        Return the new list of discounted and normalized rewards.
        """
        pass

