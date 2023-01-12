""" MAPPO : Multi-Agent Proximal Policy Optimization for DeepHive. """
import numpy as np
import torch 
import torch.nn as nn
from torch.distributions import normal, MultivariateNormal

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
        
    def linear_decay(self, dist):
            return self.std_min + (self.std_max - self.std_min) * dist

    def exponential_decay(self, dist):
        return self.std_min + (self.std_max - self.std_min) * np.exp(-dist)

    def set_std(self, dist=None):
        if self.fixed_std:
            self.action_std =  torch.full((1,), self.init_std * self.init_std).to(device)
        else:
            if self.std_type == 'linear':
                self.action_std =  torch.from_numpy(np.square(self.linear_decay(dist))).to(device)
            elif self.std_type == 'exponential':
                self.action_std =  torch.from_numpy(np.square(self.exponential_decay(dist))).to(device)
            else:
                self.action_std =  torch.full((1,), self.init_std * self.init_std).to(device)

    def decay_fixed_std(self, action_std_decay_rate=0.001):
        self.init_std = self.init_std - action_std_decay_rate
        self.init_std = round(self.init_std, 4)

        if (self.init_std <= self.std_min):
            self.init_std = self.std_min
            print(
                "setting actor output action_std to min_action_std : ", self.init_std)
        else:
            pass
            print("setting actor output action_std to : ", self.init_std)

## Create a class for the actor-critic network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim,  hidden_dim, episode_length,
                init_std=0.02, std_min=0.001, std_max=0.2, std_type='linear', fixed_std=True):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential()
        self.critic = nn.Sequential()
        for i in range(len(hidden_dim)):
            if i == 0:
                self.actor.add_module('actor_fc_{}'.format(i), nn.Linear(state_dim, hidden_dim[i]))
                self.critic.add_module('critic_fc_{}'.format(i), nn.Linear(state_dim, hidden_dim[i]))
            else:
                self.actor.add_module('actor_fc_{}'.format(i), nn.Linear(hidden_dim[i-1], hidden_dim[i]))
                self.critic.add_module('critic_fc_{}'.format(i), nn.Linear(hidden_dim[i-1], hidden_dim[i]))
            # last layer is linear with no activation
            if i == len(hidden_dim) - 1:
                self.actor.add_module('actor_fc_{}'.format(i+1), nn.Linear(hidden_dim[i], action_dim))
                self.critic.add_module('critic_fc_{}'.format(i+1), nn.Linear(hidden_dim[i], 1))

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

    def act(self, state, std_obs):
        action_mean = self.actor(state)
        # reshape action mean to be compatible with action_var
        self.std.set_std(std_obs)
        action_var = self.std.action_std
        if action_var.shape[0] == 1:
            dist = MultivariateNormal(action_mean, torch.diag(action_var))
        else:
            if self.action_dim == 1:
                action_mean = action_mean.reshape(action_var.shape)
            dist = normal.Normal(action_mean, action_var)
        action = dist.sample()
        # print("action mean : ", action_mean)
        # print("action var : ", action_var)
        # print("action : ", action)
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action, std_obs):
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)
        action_mean = self.actor(state)
        self.std.set_std(std_obs)
        action_var = self.std.action_std
        if action_var.shape[0] == 1:
            dist = MultivariateNormal(action_mean, torch.diag(action_var))
        else:
            if self.action_dim == 1:
                action_mean = action_mean.reshape(action_var.shape)
            dist = normal.Normal(action_mean, action_var)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy

## Create a class for the MAPPO agent
class MAPPO: 
    def __init__(self, state_dim, action_dim, episode_length,  
                        init_std = 0.2, std_min=0.001, std_max=0.2, std_type='linear', 
                        fixed_std=True, hidden_dim=[32,32], lr=1e-5, betas=0.99, gamma=0.9, 
                        K_epochs=32, eps_clip=0.2, initialization=None,pretrained=False, 
                        ckpt_path=None, split_agent=False, split_fraq=0.5,
                        explore_state_dim=None, exploit_state_dim=None,device=device):
        
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
        self.action_dim = action_dim
        if self.split_agent:
            self.exploration_buffer = Memory()
            self.exploitation_buffer = Memory()
        else:
            self.buffer = Memory()
        
        # create two actor-critic networks if split_agent is True (one for exploration and one for exploitation)
        # state_dim would reduce by split_fraq if split_agent is True and also n_agents would reduce by split_fraq

        if self.split_agent:
            # exploration Policy
            self.exploration_policy = ActorCritic(explore_state_dim, action_dim, 
                                                hidden_dim, self.episode_length,
                                                init_std=init_std, std_min =std_min, 
                                                std_max=std_max, std_type=std_type, 
                                                fixed_std=fixed_std).to(self.device)
            
            # exploitation Policy
            self.exploitation_policy = ActorCritic(exploit_state_dim, action_dim, 
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
            self.exploration_old_policy = ActorCritic(explore_state_dim, action_dim,
                                                    hidden_dim, self.episode_length,
                                                    init_std=init_std, std_min =std_min,
                                                    std_max=std_max, std_type=std_type,
                                                    fixed_std=fixed_std).to(self.device)

            self.exploitation_old_policy = ActorCritic(exploit_state_dim, action_dim,
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

    def select_action(self, state:np.array, std_obs:np.array, exploitation_agents_idx=[]):
        action = torch.zeros(state.shape[0], self.action_dim).to(self.device)
        action_logprob = torch.zeros(state.shape[0], self.action_dim).to(self.device)
        
        action_logprob = action_logprob.double()
        if self.split_agent:
            # select action for exploration agents
            exploration_state = state[~exploitation_agents_idx]
            exploration_std_obs = std_obs[~exploitation_agents_idx]
            exploration_state = torch.FloatTensor(exploration_state).to(self.device)
            # print("Exploration Act")
            exploration_action, exploration_action_logprob = self.exploration_policy.act(exploration_state, exploration_std_obs)

            # select action for exploitation agents
            exploitation_state = state[exploitation_agents_idx]
            exploitation_std_obs = std_obs[exploitation_agents_idx]
            exploitation_state = torch.FloatTensor(exploitation_state).to(self.device)
            # print("Exploitation Act")
            exploitation_action, exploitation_action_logprob = self.exploitation_policy.act(exploitation_state, exploitation_std_obs)
            # concatenate the actions and logprobs based on exploration_agents_idx
            action[exploitation_agents_idx] = exploitation_action.reshape(-1, self.action_dim)
            action[np.logical_not(np.isin(range(len(action)), exploitation_agents_idx))] = exploration_action.reshape(-1, self.action_dim)
            # set action_logprob type to that of exploration_action_logprob
            # set action_logprob type to that of exploration_action_logprob

            action_logprob[exploitation_agents_idx] = exploitation_action_logprob.reshape(-1, self.action_dim).double()
            action_logprob[np.logical_not(np.isin(range(len(action)), exploitation_agents_idx))] = exploration_action_logprob.reshape(-1, self.action_dim).double()
            
            for i in range(exploitation_action.shape[0]):
                self.exploitation_buffer.states.append(exploitation_state[i])
                self.exploitation_buffer.actions.append(exploitation_action[i])
                self.exploitation_buffer.logprobs.append(exploitation_action_logprob[i])
                self.exploitation_buffer.iters.append(iter)
                self.exploitation_buffer.std_obs.append(exploration_std_obs[i])
            for i in range(exploration_action.shape[0]):
                self.exploration_buffer.states.append(exploration_state[i])
                self.exploration_buffer.actions.append(exploration_action[i])
                self.exploration_buffer.logprobs.append(exploration_action_logprob[i])
                self.exploration_buffer.iters.append(iter)
                self.exploration_buffer.std_obs.append(exploitation_std_obs[i])
        else:
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob = self.policy.act(state, std_obs)

            for i in range(action.shape[0]):
                self.buffer.states.append(state[i])
                self.buffer.actions.append(action[i])
                self.buffer.logprobs.append(action_logprob[i])
                self.buffer.iters.append(iter)

        return action.detach().cpu().numpy().flatten()

        

    def __get_buffer_info(self, buffer):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(buffer.rewards), reversed(buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        old_states = torch.stack(buffer.states).to(self.device).detach()
        old_actions = torch.stack(buffer.actions).to(self.device).detach()
        old_logprobs = torch.stack(buffer.logprobs).to(self.device).detach()
        old_iters = np.stack(buffer.iters)
        if self.split_agent:
            old_std_obs = np.stack(buffer.std_obs)
        else:
            old_std_obs = buffer.std_obs
            

        return rewards, old_states, old_actions, old_logprobs, old_iters, old_std_obs


    def __update_old_policy(self, policy, old_policy, optimizer, rewards, old_states, old_actions, old_logprobs, old_std_obs):
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = policy.evaluate(old_states, old_actions,old_std_obs)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MSE_loss(state_values, rewards) - 0.001*dist_entropy

            # take gradient step
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
        
        # Copy new weights into old policy:
        old_policy.load_state_dict(policy.state_dict())
    

    def update(self):
        if self.split_agent:
            # update exploration policy
            rewards, old_states, old_actions, old_logprobs, old_iters, old_std_obs = self.__get_buffer_info(self.exploration_buffer)
            self.__update_old_policy(self.exploration_policy, self.exploration_old_policy, self.exploration_optimizer, rewards, old_states, old_actions, old_logprobs, old_std_obs)
            self.exploration_buffer.clear_memory()
            # update exploitation policy
            rewards, old_states, old_actions, old_logprobs, old_iters, old_std_obs = self.__get_buffer_info(self.exploitation_buffer)
            self.__update_old_policy(self.exploitation_policy, self.exploitation_old_policy, self.exploitation_optimizer, rewards, old_states, old_actions, old_logprobs, old_std_obs)
            self.exploitation_buffer.clear_memory()
        else:
            rewards, old_states, old_actions, old_logprobs, old_iters, old_std_obs = self.__get_buffer_info(self.buffer)
            self.__update_old_policy(self.policy, self.old_policy, self.optimizer, rewards, old_states, old_actions, old_logprobs, old_std_obs)
            self.buffer.clear_memory()

            
    def save(self, filename):
        if self.split_agent:
            torch.save(self.exploration_policy.state_dict(), filename + "_exploration" + ".pth")
            torch.save(self.exploitation_policy.state_dict(), filename + "_exploitation"  + ".pth")
            print("Saved exploration policy to: ", filename + "_exploration")
            print("Saved exploitation policy to: ", filename + "_exploitation")
        else:
            torch.save(self.policy.state_dict(), filename + ".pth")
            print("Saved policy to: ", filename)

    def load(self, filename):
        if self.split_agent:
            self.exploration_policy.load_state_dict(torch.load(filename + "_exploration" + ".pth"))
            self.exploitation_policy.load_state_dict(torch.load(filename + "_exploitation" + ".pth"))
            print("Loaded exploration policy from: ", filename + "_exploration")
            print("Loaded exploitation policy from: ", filename + "_exploitation")
        else:
            self.policy.load_state_dict(torch.load(filename + ".pth"))
            print("Loaded policy from: ", filename) 