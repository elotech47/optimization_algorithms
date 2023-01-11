""" 
DeepHive environment: 
    - The environment is a multi-agent environment where each agent is a particle in a swarm.
    - The agents are initialized randomly in the search space.
    - The agents are updated according to the DeepHive algorithm.
    - The agents are rewarded based on the objective function.
    - The agents are terminated when the episode length is reached.
"""
from gym.utils import seeding
from gym import spaces
from collections import deque
from typing import Optional
import gym
import numpy as np
from typing import Callable, List, Tuple, Optional
import matplotlib.pyplot as plt
import pickle

class OptEnv(gym.Env):
    def __init__(self, env_name, optFunc:Callable, n_agents:int, n_dim:int, bounds, ep_length=20, minimize=False, freeze=True, init_state=None, fraq_refinement=0.5, opt_value=None)->None:
        """
        Args:
            env_name: the name of the environment
            optFunc: the function to be optimized
            n_agents: number of agents
            n_dim: number of dimensions 
            bounds: the bounds of the function [ (min_bounds), (max_bounds)]
            ep_length: the length of the episode
            minimize: whether to minimize the function
            freeze: whether to freeze the best agent
            init_state: the initial state of the agents [List of np.array]
            opt_bound: the bound of the optimal value
            reward_type: the type of the reward (Check readme for more details)
            fraq_refinement: the fraction of the agents to be used for refinement (exploitation)
            opt_value: the optimal value of the function
            
            **kwargs: other arguments
        """
        super(OptEnv, self).__init__()
        self.env_name = env_name
        self.optFunc = optFunc
        self.n_agents = n_agents
        self.n_dim = n_dim
        self.min_pos = np.array(bounds[0])
        self.max_pos = np.array(bounds[1])
        self.ep_length = ep_length
        self.init_state = init_state 
        self.freeze = freeze
        self.refinement_idx = []
        self.fraq_refinement = fraq_refinement
        self.minimize = minimize
        self.done = [False] * self.n_agents
        # check if opt_func is not none but it can be zero
        if opt_value is None:
            print("Optimal value is not provided. The optimal value will be calculated.")
            self.opt_obj_value = -np.inf if self.minimize else np.inf
        else:
            print("Optimal value is provided and set to: ", opt_value, ".")
            self.opt_obj_value = opt_value


        self.lower_bound_actions = np.array(
            [-np.inf for _ in range(self.n_dim)], dtype=np.float64) # lower bound of the action space
        self.upper_bound_actions = np.array(
            [np.inf for _ in range(self.n_dim)], dtype=np.float64) # upper bound of the action space
        self.lower_bound_obs = np.append(np.array(
            self.min_pos, dtype=np.float64), -np.inf) # lower bound of the observation space
        self.upper_bound_obs = np.append(np.array(
            self.max_pos, dtype=np.float64), np.inf) # upper bound of the observation space
        
        self.obs_low = np.array([self.lower_bound_obs for _ in range(self.n_agents)])  # lower bound of the observation space for all agents
        self.obs_high = np.array([self.upper_bound_obs for _ in range(self.n_agents)]) # upper bound of the observation space for all agents

        self.action_low = np.array([self.lower_bound_actions for _ in range(self.n_agents)]) # lower bound of the action space for all agents
        self.action_high = np.array([self.upper_bound_actions for _ in range(self.n_agents)]) # upper bound of the action space for all agents

        self.action_space = spaces.Box(
            low=self.action_low, high=self.action_high, shape=(
                 self.n_agents,self.n_dim), dtype=np.float64) # gym action space
        self.observation_space = spaces.Box(
            self.obs_low, self.obs_high, dtype=np.float64) # gym observation space
    
    def step(self, actions:np.ndarray)->Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Args:
            actions: the actions taken by the agents
        Returns:
            obs: the observation of the agents
            rewards: the rewards of the agents
            done: whether the episode is done
            info: additional information
        """
        assert self.action_space.contains(actions), f"{actions!r} ({type(actions)}) invalid" # check if the action is valid

        states = self.state.copy()  # get the current state
        bestAgent = np.argmin(self.state[:,-1]) if self.minimize else np.argmax(self.state[:,-1]) # get the best agent
        zprev = states.copy() # store the previous state
        states[:, :-1] += actions # update the state
        if self.freeze:
            states[bestAgent, :-1] = zprev[bestAgent, :-1] # freeze the best agent
        states[:, :-1] = np.clip(states[:, :-1], 0, 1) # clip the state to be in [0,1]
        self.obj_values = self.optFunc(self._rescale(states[:, :-1], self.min_pos, self.max_pos), self.minimize) # get the objective values
        # add obj_values to ValueHistory for each step
        self.ValueHistory[self.current_step+1, :] = self.obj_values
        
        if self.current_step >= self.ep_length-1:
            # set the done list to true for all agents
            self.done = [True for _ in range(self.n_agents)]
        
        # check which agents have 90% of the opt_value and set their done to True
        if self.minimize:
            self.done = np.where(self.obj_values <= (self.opt_obj_value + 0.001), True, False)
        else:
            self.done = np.where(self.obj_values >= 0.7*self.opt_obj_value, True, False)

        # print("Step: ", self.current_step, " Obj_values: ", self.obj_values, " Done: ", self.done, "Optimal value: ", self.opt_obj_value)

        # add self.done to all agents without for loop
        agents_done = np.array(self.done)
        self.current_step += 1  # Increment step counter
        
        # update best objective value
        if self.minimize:
            if np.min(self.obj_values) < self.best_agent_value:
                self._set_episode_best_info()
        else:
            if np.max(self.obj_values) > self.best_agent_value:
                self._set_episode_best_info()

        # scale objective value to [0, 1]
        states[:, -1] = self._scale(self.obj_values, self.worst_agent_value, self.best_agent_value)
        self.state = states
        
        # store StateHistory for each step
        self.stateHistory[self.current_step, :, :] = self._get_actual_state()
        # store ValueHistory for each step
        self.ValueHistory[self.current_step, :] = self.obj_values
        self.refinement_idx = self._get_refinement_idxs()
        self.best_agent_idxs.append(self.best_agent_idx)
        # get the reward for each agent
        rewards = self._reward_fn(bestAgent)
        # update episode return
        self.episode_return += rewards
        return self.state, rewards, agents_done, self.obj_values

    def _scale(self, x, min, max):
        """ Scale x to [0, 1]"""
        return (x - min) / ((max - min) + 1e-8)

    def _rescale(self, x, min, max):
        """ Rescale x from [0, 1] to [min, max]"""
        return x * (max - min) + min

    def _measure_reward(self, z, best=1):
        return 20 * (z - 0.5)

    def _reward_fn(self, best_idx):
        reward =  self._measure_reward(self.state[:, -1])
        # set the reward of the best agent to 0
        reward[best_idx] += 10
        # if agent is stock for a long time, give it a negative reward
        for agent in range(self.n_agents):
            z_recent = self.ValueHistory[agent][-min(3, self.current_step):]
            if len(np.unique(z_recent)) == 1 and self.current_step > 3 and np.unique(z_recent) != 0:
                reward[agent] -= 5
            # add a reward if agent is at 90% close to the opt_value
            if self.minimize:
                if self.obj_values[agent] <= self.opt_obj_value + 0.001:
                    reward[agent] += 100 * ((self.opt_obj_value + 0.0001) / self.obj_values[agent])
            else:
                if self.obj_values[agent] >= 0.5*self.opt_obj_value:
                    reward[agent] += (self.obj_values[agent] / self.opt_obj_value) * 100
            #print(f"Agent {agent} reward: {reward[agent]}, obj_value: {self.obj_values[agent]}")
        return reward/(self.current_step+1 * 0.5) 
            
    def _generate_init_state(self):
        if self.init_state is None:
            init_pos = np.round(np.random.uniform(low=self.obs_low[0][:-1], high=self.obs_high[0][:-1],), decimals=2)
        else:
            init_pos = np.array(self.init_state)
        # generate a random initial position for all agents at once
        init_pos = np.round(np.random.uniform(low=self.obs_low[0][:-1], high=self.obs_high[0][:-1], size=(self.n_agents, self.n_dim)), decimals=2)
        # get the objective value of the initial position
        init_obj = self.optFunc(init_pos, self.minimize)
        # scale the position to [0, 1]
        init_pos = self._scale(init_pos, self.min_pos, self.max_pos)
        # combine the position and objective value
        init_obs = np.append(init_pos, init_obj.reshape(-1, 1), axis=1)
        return init_obs, init_obj
    
    def _get_refinement_idxs(self):
        if self.minimize:
            sorted_idx = np.argsort(np.linalg.norm(self.state[:, :2] - self.state[np.argmin(self.state[:, -1]), :2], axis=1))
        else:
            sorted_idx = np.argsort(np.linalg.norm(self.state[:, :2] - self.state[np.argmax(self.state[:, -1]), :2], axis=1))
        n = int(self.fraq_refinement * self.n_agents)
        self.refinement_idxs.append(sorted_idx[:n])
        return sorted_idx[:n]

    def _set_episode_best_info(self):
        """ Set the best agent and best objective value for the current episode"""
        sorted_obj = sorted(self.obj_values, reverse=not self.minimize)
        self.best_agent_idx = np.where(self.obj_values == sorted_obj[0])[0][0]
        # self.best_agent_idxs.append(self.best_agent_idx)
        self.worst_agent_idx = np.where(self.obj_values == sorted_obj[-1])[0][0]
        self.best_time_step = self.current_step  # store the time step when the best objective value is found
        self.best_agent_value = self.obj_values[self.best_agent_idx]
        self.worst_agent_value = self.obj_values[self.worst_agent_idx]
        # get the best agent but in the scaled value
        self.best_agent = self.state[self.best_agent_idx].copy()
        self.best_agent[-1] = self._scale(self.obj_values[self.best_agent_idx], self.worst_agent_value, self.best_agent_value)
        self.worst_agent = self.state[self.worst_agent_idx].copy()
        self.worst_agent[-1] = self._scale(self.obj_values[self.worst_agent_idx], self.worst_agent_value, self.best_agent_value)


    def _get_actual_state(self):
        actual_state = self._rescale(self.state[:, :2], self.min_pos, self.max_pos)
        # append the obj_value to the actual state
        actual_state = np.append(actual_state, self.obj_values.reshape(-1, 1), axis=1)
        return actual_state

    def reset(self, seed: Optional[int] = None):
        #super().reset()
        self.current_step = 0
        self.state, self.obj_values = self._generate_init_state() #np.array([self._generate_init_state(agent) for agent in range(self.n_agents)])
         # store the state history of each agent
        self.episode_return = np.zeros(shape=(1, self.n_agents))
        self.bestValueHistory = np.zeros(shape=(self.ep_length, 1))
        self.ValueHistory = np.zeros(shape=(self.ep_length+1, self.n_agents)) # store the objective value history of each agent
        self.stateHistory = np.zeros(shape=(self.ep_length+1, self.n_agents,  self.n_dim+1))
        self.refinement_idxs = []
        self.best_agent_idxs = []
        self._set_episode_best_info()
        self.ValueHistory[self.current_step, :] = self.obj_values
        self.bestValueHistory[self.current_step] = self.best_agent_value
        self.stateHistory[self.current_step, :, :] = self._get_actual_state()
        self.state[:,-1] = self._scale(self.state[:,-1], self.worst_agent_value, self.best_agent_value)
        self.refinement_idx = self._get_refinement_idxs()
        return np.array(self.state, dtype=np.float32)

    def _validate_state(self):
        # check if self.state values is within range 0-1
        if np.any(self.state < 0) or np.any(self.state > 1):
            raise ValueError("State values must be within [0, 1]")

    ## get best agent actual position and value
    def get_best_agent(self):
        return self.stateHistory[self.best_time_step][self.best_agent_idx]
    
    def render(self):
        # render the environment to the screen if self.n_dim is <= 2
        if self.n_dim == 2:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.scatter(self.state[:, 0], self.state[:, 1], c=self.state[:, -1], cmap='viridis', s=100)
            plt.show(block=False)
            #plt.close()

        elif self.n_dim == 1:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.scatter(self.state[:, 0], np.zeros(self.n_agents), c=self.state[:, -1], cmap='viridis')
            #plt.show()
            plt.close()
        else:
            raise NotImplementedError

    def close(self):
        # close the environment
        pass

# create a class to cache different environments for different objective functions
class OptEnvCache:
    def __init__(self):
        self.envs = {}

    def get_env(self, env_name):
        if env_name in self.envs:
            return self.envs[env_name]
        else:
            raise ValueError("Environment not found")
    
    def add_env(self, env_name, env):
        # check if environment is valid
        if not isinstance(env, OptEnv):
            raise ValueError("Environment is not valid")
        # check if the environment already exists
        if env_name in self.envs:
            raise ValueError("Environment already exists")
        else:
            self.envs[env_name] = env

    def remove_env(self, env_name):
        if env_name in self.envs:
            del self.envs[env_name]
        else:
            raise ValueError("Environment not found")

    def get_env_names(self):
        return self.envs.keys()

    def get_envs(self):
        return self.envs.values()

    def num_envs(self):
        return len(self.envs)

    def clear(self):
        self.envs = {}
    
    def save_envs(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.envs, f)
    

if __name__ == "__main__":
    n_agents = 6
    n_dim = 2
    bounds = [(0, 0), (1,1)] 
    ep_length = 10

    def sphere_func_2d(x):
        return np.sum(x**2)
    
    env = OptEnv(sphere_func_2d, n_agents, n_dim, bounds, ep_length, minimize=True)
    states = env.reset()
    for i in range(ep_length):
        print(env.step(np.random.uniform(low=-1, high=1, size=(6, 2))))
        env.render()
        plt.close()
