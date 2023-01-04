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
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.std_obs = []
        self.iters = []
    
    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.std_obs[:]
        del self.iters[:]

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
        self.std = StdDecay(self.init_std, self.std_min, self.std_max, self.std_type, self.fixed_std)
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
    def __init__(self, n_agents, n_dim, state_dim, action_dim, episode_length,  
                        init_std = 0.2, std_min=0.001, std_max=0.2, std_type='linear', 
                        fixed_std=True, hidden_dim=[32,32], lr=1e-5, betas=0.99, gamma=0.9, 
                        K_epochs=32, eps_clip=0.2, initialization=None,pretrained=False, 
                        ckpt_path=None, device=device, split_agent=False, split_fraq=0.5):
        
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.episode_length = episode_length
        self.device = device
        self.pretrained = pretrained
        self.ckpt_path = ckpt_path
        self.initialization = initialization
        self.split_agent = split_agent
        if self.split_agent:
            self.exploration_buffer = Memory()
            self.exploitation_buffer = Memory()
        else:
            self.buffer = Memory()

        # create two actor-critic networks if split_agent is True (one for exploration and one for exploitation)
        # state_dim would reduce by split_fraq if split_agent is True and also n_agents would reduce by split_fraq

        if self.split_agent:
            # exploration Policy
            exploration_state_dim = int(state_dim * split_fraq)
            exploitation_state_dim = state_dim - exploration_state_dim

            self.exploration_policy = ActorCritic(exploration_state_dim, action_dim, 
                                                hidden_dim, self.episode_length,
                                                init_std=init_std, std_min =std_min, 
                                                std_max=std_max, std_type=std_type, 
                                                fixed_std=fixed_std).to(self.device)
            
            # exploitation Policy
            self.exploitation_policy = ActorCritic(exploitation_state_dim, action_dim, 
                                                hidden_dim, self.episode_length,
                                                init_std=init_std, std_min =std_min, 
                                                std_max=std_max, std_type=std_type, 
                                                fixed_std=fixed_std).to(self.device)

            if self.pretrained:
                self.exploration_policy.load_state_dict(torch.load(self.ckpt_path + 'exploration_policy.pt'))
                self.exploitation_policy.load_state_dict(torch.load(self.ckpt_path + 'exploitation_policy.pt'))

            self.exploration_optimizer = torch.optim.Adam(self.exploration_policy.parameters(), lr=self.lr, betas=(self.betas, 0.999))
            self.exploitation_optimizer = torch.optim.Adam(self.exploitation_policy.parameters(), lr=self.lr, betas=(self.betas, 0.999))

            # old policy
            self.exploration_old_policy = ActorCritic(exploration_state_dim, action_dim,
                                                    hidden_dim, self.episode_length,
                                                    init_std=init_std, std_min =std_min,
                                                    std_max=std_max, std_type=std_type,
                                                    fixed_std=fixed_std).to(self.device)

            self.exploitation_old_policy = ActorCritic(exploitation_state_dim, action_dim,
                                                    hidden_dim, self.episode_length,    
                                                    init_std=init_std, std_min =std_min,
                                                    std_max=std_max, std_type=std_type, 
                                                    fixed_std=fixed_std).to(self.device)

            self.exploration_old_policy.load_state_dict(self.exploration_policy.state_dict())
            self.exploitation_old_policy.load_state_dict(self.exploitation_policy.state_dict())


        else:
            self.policy = ActorCritic(state_dim, action_dim, hidden_dim, self.episode_length,
                                    init_std=init_std, std_min =std_min, std_max=std_max, 
                                    std_type=std_type, fixed_std=fixed_std).to(self.device)
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=(self.betas, 0.999))

            # old policy
            self.old_policy = ActorCritic(state_dim, action_dim, hidden_dim, self.episode_length,
                                        init_std=init_std, std_min =std_min, std_max=std_max, 
                                        std_type=std_type, fixed_std=fixed_std).to(self.device)

            self.old_policy.load_state_dict(self.policy.state_dict())

        self.MSE_loss = nn.MSELoss()

    def select_action(self, state, std_obs, iter, exploration_agents_idx):
        if self.split_agent:
            # select action for exploration agents
            exploration_state = state[exploration_agents_idx]
            exploration_state = torch.FloatTensor(exploration_state).to(self.device)
            exploration_action, exploration_action_logprob = self.exploration_policy.act(exploration_state, std_obs, iter)

            # select action for exploitation agents
            exploitation_state = state[ ~exploration_agents_idx]
            exploitation_state = torch.FloatTensor(exploitation_state).to(self.device)
            exploitation_action, exploitation_action_logprob = self.exploitation_policy.act(exploitation_state, std_obs, iter)

            # concatenate the actions
            action = torch.cat((exploration_action, exploitation_action), dim=1)
            action_logprob = torch.cat((exploration_action_logprob, exploitation_action_logprob), dim=1)

            # for i in range(action.shape[0]):
            #     if i in exploration_agents_idx:
            #         self.exploration_buffer.states.append(state[i])
            #         self.exploration_buffer.actions.append(action[i])
            #         self.exploration_buffer.logprobs.append(action_logprob[i])
            #         self.exploitation_buffer.iters.append(iter)
            #     else:
            #         self.exploitation_buffer.states.append(state[i])
            #         self.exploitation_buffer.actions.append(action[i])
            #         self.exploitation_buffer.logprobs.append(action_logprob[i])
            # populating the buffer without for loop
            self.exploration_buffer.states.append(state[exploration_agents_idx])
            self.exploration_buffer.actions.append(action[exploration_agents_idx])
            self.exploration_buffer.logprobs.append(action_logprob[exploration_agents_idx])
            self.exploration_buffer.iters.append(iter)
            

        else:
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob = self.policy.act(state, std_obs, iter)

        return action, action_logprob

