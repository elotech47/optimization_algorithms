""" 
Unit tests for the deephive package.
1. First test coverage is the environment is set up correctly.
"""

import unittest
import numpy as np
import os
import sys
import time
from algorithms.deephive.environment import OptEnv


class TestOptEnv(unittest.TestCase):
    def setUp(self):
        self.n_agents = 6
        self.n_dim = 2
        self.bounds = [(0, 0), (1,1)] 
        self.ep_length = 10

        # Define objective function
        def sphere_func(x):
            return -np.sum(x**2, axis=1)
        
        self.env = OptEnv(sphere_func, self.n_agents, self.n_dim, self.bounds, self.ep_length, minimize=True)

    def test_env_setup(self):
        self.assertEqual(self.env.n_agents, self.n_agents)
        self.assertEqual(self.env.n_dim, self.n_dim)
        self.assertEqual(self.env.ep_length, self.ep_length)
        self.assertEqual(self.env.minimize, True)

    def test_env_reset(self):
        self.env.reset()
        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(self.env.state.shape, (self.n_agents, self.n_dim + 1))
        # self.assertEqual(self.env.best, np.max(self.env.state[:, -1]))
        # self.assertEqual(self.env.worst, np.min(self.env.state[:, -1]))
        self.assertEqual(self.env.refinement_idx.shape, ((self.env.fraq_refinement * self.env.n_agents),))

    def test_env_step(self):
        self.env.reset()
        self.env.step(np.random.rand(self.n_agents, self.n_dim))
        self.assertEqual(self.env.current_step, 1)
        self.assertEqual(self.env.state.shape, (self.n_agents, self.n_dim + 1))
        self.assertEqual(self.env.refinement_idx.shape, ((self.env.fraq_refinement * self.env.n_agents),))

    def test_env_done(self):
        self.env.reset()
        for i in range(self.ep_length):
            self.env.step(np.random.rand(self.n_agents, self.n_dim))
        self.assertEqual(self.env.done, True)

if __name__ == '__main__':
    unittest.main()



