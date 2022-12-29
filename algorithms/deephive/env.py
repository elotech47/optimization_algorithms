from gym.utils import seeding
from gym import spaces
from collections import deque
from typing import Optional
import gym
import numpy as np
from typing import Callable, List, Tuple, Optional
import matplotlib.pyplot as plt

class OptEnv(gym.Env):
    def __init__(self, optFunc:Callable, n_agents:int, n_dim:int, bounds, ep_length:int, minimize=False, freeze=False, init_state=None, opt_bound=0.9, reward_type=0, fraq_refinement=0.5)->None:
        """
        Args:
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
            
            **kwargs: other arguments
        """
        self.optFunc = optFunc
        self.n_agents = n_agents
        self.n_dim = n_dim
        self.min_pos = np.array(bounds[0])
        self.max_pos = np.array(bounds[1])
        self.ep_length = ep_length
        self.init_state = init_state 
        self.opt_bound = opt_bound
        self.reward_type = reward_type
        self.freeze = freeze
        self.refinement_idx = []
        self.fraq_refinement = fraq_refinement
        self.minimize = minimize

        self.stateHistory = [deque(maxlen=self.ep_length) for _ in range(self.n_agents)] # store the state history of each agent
        self.ValueHistory = [[] for _ in range(self.n_agents)] # store the value history of each agent
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
                 self.n_agents,self.n_dim), dtype=np.float64
        ) # gym action space
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

        agents_done = []
        states = self.state  # get the current state
        obj_values = []
        bestAgent = np.argmax(self.state[:,-1]) # get the best agent
        #zprev = states[:, -1].copy() # store the previous state
        for agent in range(self.n_agents):
            # Do not update the best agent if freeze is True
            if agent == bestAgent and self.freeze:
                pass
            else:
                for i in range(self.n_dim):
                    # Update the state:
                    states[agent][i] += actions[agent][i]
            #Retrict the state to the bounds:
            states[agent] = np.clip(states[agent], 0, 1)
            # Get objective value:
            ## rescale the state:
            obj_v = self.optFunc(self._rescale(states[agent][:-1], self.min_pos, self.max_pos))
            self.stateHistory[agent].append(states[agent])
            self.ValueHistory[agent].append(obj_v)
            obj_values.append(obj_v)
 
        done = False
        if self.current_step >= self.ep_length-1:
            done = True
        for agent in range(self.n_agents):
            agents_done.append(done)
        
        # update best objective value
        if self.minimize:
            if np.min(obj_values) <= self.best:
                self.best = np.min(obj_values)
            if np.max(obj_values) > self.worst:
                self.worst = np.max(obj_values)
                #print("Worst: ", self.worst)
        else:
            if np.max(obj_values) >= self.best:
                self.best = np.max(obj_values)
            if np.min(obj_values) < self.worst:
                self.worst = np.min(obj_values)

        # scale objective value to [0, 1]
        states[:, -1] = self._scale(obj_values, self.worst, self.best)
        self.state = states
        self.refinement_idx = self._get_refinement_idxs()
        # get the reward for each agent
        rewards = self._reward_fn(bestAgent)
        self.current_step += 1  # Increment step counter
        
        return self.state, rewards, agents_done, obj_values

    def _scale(self, x, min, max):
        """ Scale x to [0, 1]"""
        return (x - min) / (max - min)

    def _rescale(self, x, min, max):
        """ Rescale x from [0, 1] to [min, max]"""
        return x * (max - min) + min

    def _measure_reward(self, z, best=1):
        return 10*np.exp(abs(z - best)**2)    

    def _reward_fn(self, bestAgent):
        reward =  self._measure_reward(self.state[:, -1], self.state[bestAgent, -1])
        # set the reward of the best agent to 0
        reward[bestAgent] = 0
        # if agent is stock for a long time, give it a negative reward
        for agent in range(self.n_agents):
            z_recent = self.ValueHistory[agent][-min(3, self.current_step):]
            if len(np.unique(z_recent)) == 1 and self.current_step > 3:
                reward[agent] = -1
        return reward
            
    def _generate_init_state(self, agent):
        if self.init_state is None:
            init_pos = np.round(np.random.uniform(low=self.obs_low[0][:-1], high=self.obs_high[0][:-1],), decimals=2)
        else:
            init_pos = np.array(self.init_state[agent])

        #print("Initial position: ", init_pos)
        init_obj = self.optFunc(init_pos)
        #print("Initial objective: ", init_obj)
        init_pos = self._scale(init_pos, self.min_pos, self.max_pos)
        init_obs = np.append(init_pos, init_obj)
        return init_obs
    
    def _get_refinement_idxs(self):
        sorted_idx = np.argsort(self.state[:, -1])[::-1]
        print("Sorted idx: ", sorted_idx)
        print("Fraq refinement: ", self.fraq_refinement)
        print("N agents: ", self.n_agents)
        n = int(self.fraq_refinement * self.n_agents)
        return sorted_idx[:n]

    def reset(self, seed: Optional[int] = None):
        #super().reset()
        self.current_step = 0
        self.state = np.array([self._generate_init_state(agent) for agent in range(self.n_agents)])
        self.best = np.max(self.state[:, -1])
        self.worst = np.min(self.state[:, -1])
        self.state[:,-1] = self._scale(self.state[:,-1], self.worst, self.best)
        print("Initial state value: ", self.state[:,-1])
        self.refinement_idx = self._get_refinement_idxs()
        return np.array(self.state, dtype=np.float32)


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

