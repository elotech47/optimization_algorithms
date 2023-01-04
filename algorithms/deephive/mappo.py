""" MAPPO : Multi-Agent Proximal Policy Optimization for DeepHive. """
import os
import numpy as np
import time
import torch 
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal, Normal
import math
from datetime import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random_seed = 42

## Create a buffer class to store the trajectories
class Buffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.std_obs = []
    
    def clear_buffer(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.std_obs[:]

## Create a class for std decay
class StdDecay:
    def __init__(self, init_std=0.02, std_min=0.001, std_max=0.2, std_type='linear', fixed_std=True):
        self.init_std = init_std
        self.std_min = std_min
        self.std_max = std_max
        self.std_type = std_type
        self.fixed_std = fixed_std

    def get_std(self, dist, iter, max_iter):
        if self.fixed_std:
            return self.init_std
        else:
            if self.std_type == 'linear':
                return max(self.std_min, self.std_max - (self.std_max - self.std_min) * iter / max_iter) * dist
            elif self.std_type == 'exponential':
                return max(self.std_min, self.std_max * (self.std_min / self.std_max) ** (iter / max_iter)) * dist
            else:
                raise NotImplementedError

## Create a class for the actor-critic network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim,  hidden_dim, episode_length,
                init_std=0.02, std_min=0.001, std_max=0.2, std_type='linear', fixed_std=True):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.action_dim = action_dim
        self.std_min = std_min
        self.std_max = std_max
        self.std_type = std_type
        self.fixed_std = fixed_std
        self.init_std = init_std
        self.std = StdDecay(init_std, std_min, std_max, std_type, fixed_std)
        self.episode_length = episode_length
        

    def forward(self):
        raise NotImplementedError

    def act(self, state, std_obs, iter):
        action_mean = self.actor(state)
        action_var = self.std.get_std(std_obs, iter, self.episode_length)
        dist = Normal(action_mean, action_var)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob

    def evaluate(self, state, action, std_obs, iter):
        action_mean = self.actor(state)
        action_var = self.std.get_std(std_obs, iter, self.episode_length)
        dist = Normal(action_mean, action_var)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy

## Create a class for the MAPPO agent
class MAPPO:
    def __init__(self, state_dim, action_dim, hidden_dim, episode_length, 
                init_std=0.02, std_min=0.001, std_max=0.2, std_type='linear', fixed_std=True,
                lr=0.001, betas=(0.9, 0.999), gamma=0.99, K_epochs=80, eps_clip=0.2, 
                num_agents=1, num_envs=1, num_steps=1, num_mini_batch=1, use_gae=True, 
                gae_lambda=0.95, use_proper_time_limits=False, value_loss_coef=0.5, 
                entropy_coef=0.01, max_grad_norm=0.5, use_linear_lr_decay=False, 
                use_linear_clip_decay=False, use_recurrent_policy=False, recurrent_hidden_size=64, 
                add_timestep=False, normalize_advantage=False, use_popart=False, 
                use_normalized_advantage=False, use_shared_policy=False, use_shared_value=False, 
                use_shared_std=False):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.num_agents = num_agents
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.num_mini_batch = num_mini_batch
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda