"""
DeepHive Algorithm Implementation: A multi-agent reinforcement learning algorithm for black-box optimization.

"""
#imports
import numpy as np
import random
import math
from typing import Callable, List, Tuple
from algorithm.opt_env import OptEnv, OptEnvCache
from algorithm.mappo import MAPPO
import os
from argparse import ArgumentParser
import yaml
from algorithm.optimization_functions import *

def get_args():
    """
    Get the command line arguments.
    :return: The command line arguments.
    """
    # create the parser
    parser = ArgumentParser()
    # add the arguments
    parser.add_argument('-c', '--config', type=str, default='config.yaml', help='The path to the config file.')
    # parse the arguments
    args = parser.parse_args()
    # return the arguments
    return args

def get_config(path:str) -> dict:
    """
    Get the config from the yaml file.
    :param path: The path to the yaml file.
    :return: The config.
    """
    # check if the file exists
    if not os.path.exists(path):
        raise FileNotFoundError('The config file does not exist.')
    # open the file
    with open(path, 'r') as file:
        # load the yaml file
        config = yaml.safe_load(file)
    # return the config
    return config

def prepare_environment(config:dict) -> OptEnvCache:
    """
    Prepare the environment.
    :param config: The config.
    :return: The environment.
    """
    envs_cache = OptEnvCache()
    # prepare each environment as specified in the config file and cache them
    env_config = config['environment_config']
    for i in range(len(env_config['env_list'])):
        # create the environment
        func_name = env_config['env_list'][i] + "_" + f"{env_config['n_dim']}D_{env_config['n_agents']}_agents"
        obj_func, bounds = get_obj_func(env_config['env_list'][i])
        # if n_dim is not 2D, ignore the bounds and use bounds from the config file
        if env_config['n_dim'] != 2:
            bounds = env_config['bounds'][i]
        env = OptEnv(func_name, obj_func, env_config['n_agents'], env_config['n_dim'], 
                    bounds, env_config['ep_length'], env_config['minimize'], env_config['freeze']
                    )
        # cache the environment
        envs_cache.add_env(func_name, env)
    # return the environment
    return envs_cache

def initialize_policy(config:dict, ep_length:int) -> MAPPO:
    """
    Initialize the policy.
    :param config: The config.
    :return: The policy.
    """
    # get the policy config
    policy_config = config['policy_config']
    # create the policy
    policy = MAPPO(policy_config['state_dim'], policy_config['action_dim'],ep_length,policy_config['init_std'],
                    policy_config['std_min'], policy_config['std_max'], policy_config['std_type'], policy_config['fixed_std'],
                    policy_config['hidden_dim'], policy_config['lr'], policy_config['betas'], policy_config['gamma'],
                    policy_config['K_epochs'], policy_config['eps_clip'], policy_config['initialization'],
                    policy_config['pretrained'], policy_config['ckpt_path'],policy_config['split_agent'],
                     policy_config['split_fraq'], policy_config['explore_state_dim'],policy_config['exploit_state_dim'])
    # return the policy
    return policy
def exploitation_communication_topology(states, global_best, minimize=False):
    """
    Exploitation communication topology.
    :param states: The states.
    :param global_best: The global best.
    :param minimize: Whether the objective function is minimized.
    :return: The communication topology.
    """
    # get the number of agents
    n_agents = states.shape[0]
    n_dim = states.shape[1] - 1
    # initialize the communication topology
    observation = [[] for _ in range(n_dim)]
    std_observation = [[] for _ in range(n_dim)]
    # get the distance between the agents and the global best for each dimension
    for agent in range(n_agents):
        # std = euclidean distance between the agent and the global best for dimensions except the last one
        std = np.linalg.norm(states[agent, :-1] - global_best[:-1])
        for dim in range(n_dim):
            obs = [states[agent, dim], global_best[dim], states[agent, n_dim], global_best[n_dim]]
            observation[dim].append(np.array(obs))
            std_observation[dim].append(std)

    observation_ = [np.array(observation[dim]).reshape(n_agents, len(obs)) for dim in range(n_dim)]
    std_observation_ = [np.array(std_observation[dim]).reshape(n_agents, 1) for dim in range(n_dim)]
    return observation_, std_observation_

def exploration_communication_topology(states, minimize=False):
    """ 
    Exploration communication topology.
    :param states: The states.
    :param minimize: Whether the objective function is minimized.
    :return: The communication topology.
    """
    # get the number of agents
    n_agents = states.shape[0]
    n_dim = states.shape[1] - 1
    # initialize the communication topology
    observation = [[] for _ in range(n_dim)]
    std_observation = [[] for _ in range(n_dim)]
    # get best agent among states
    best_agent = np.argmin(states[:, -1]) if minimize else np.argmax(states[:, -1])
    for agent in range(n_agents):
        agent_nbs = [i for i in range(n_agents) if i != agent]
        # get a random neighbor
        nb = np.random.choice(agent_nbs)
        std = np.linalg.norm(states[agent, :-1] - states[nb, :-1])
        for dim in range(n_dim):
            obs = [states[agent, dim], states[nb, dim], states[agent, n_dim], states[nb, n_dim]]
            observation[dim].append(np.array(obs))
            std_observation[dim].append(std)

    observation_ = [np.array(observation[dim]).reshape(n_agents, len(obs)) for dim in range(n_dim)]
    std_observation_ = [np.array(std_observation[dim]).reshape(n_agents, 1) for dim in range(n_dim)]
    return observation_, std_observation_

def general_communication_topology(states, minimize=False):
    """
    General communication topology.
    :param states: The states.
    :param minimize: Whether the objective function is minimized.
    :return: The communication topology.
    """
    # get the number of agents
    n_agents = states.shape[0]
    n_dim = states.shape[1] - 1
    # initialize the communication topology
    observation = [[] for _ in range(n_dim)]
    std_observation = [[] for _ in range(n_dim)]
    # get the distance between the agents for each dimension
    best_agent = np.argmin(states[:, -1]) if minimize else np.argmax(states[:, -1])
    for agent in range(n_agents):
        for nb in range(n_agents):
            if agent != nb:
                std = np.linalg.norm(states[agent, :-1] - states[nb, :-1])
                for dim in range(n_dim):
                    obs = [states[agent, dim], states[nb, dim], states[agent, n_dim], states[nb, n_dim],
                            states[best_agent, dim], states[best_agent, n_dim]]
                    observation[dim].append(np.array(obs))
                    std_observation[dim].append(std)

    observation_ = [np.array(observation[dim]).reshape(n_agents, len(obs)) for dim in range(n_dim)]
    std_observation_ = [np.array(std_observation[dim]).reshape(n_agents,  1) for dim in range(n_dim)]
    return observation_, std_observation_

def generate_observations(env: OptEnv, split=False) -> np.ndarray:
    """
    Generate the observations.
    :param env: The environment.
    :param split: Whether the agents are splitted for exploration and exploitation.
    :return: The observations.
    """
    # create an array as place holder for the observations
    observation = []
    std_observation = []
    states = env.state
    global_best = env.best_agent
    # if the agents are splitted, get reinfinement_idx
    if split:
        refinement_idx = env.refinement_idx
        # get the communication topology for exploitation
        exploit_observation, exploit_std_observation = exploitation_communication_topology(states, global_best, env.minimize)
        # get the communication topology for exploration
        explore_observation, explore_std_observation = exploration_communication_topology(states, env.minimize)
        # combine the both observation and std_observation based on the reinforcement_idx
        for dim in range(env.n_dim):
            obs = [None] * env.n_agents
            std_obs = [None] * env.n_agents
            for agent in range(env.n_agents):
                if agent in refinement_idx:
                    obs[agent] = exploit_observation[dim][agent].tolist()
                    std_obs[agent] = exploit_std_observation[dim][agent][0]
                else:
                    obs[agent] = explore_observation[dim][agent][:].tolist()
                    std_obs[agent] = explore_std_observation[dim][agent][0]
            observation.append(np.array(obs))
            std_observation.append(np.array(std_obs))
    else:
        # get the communication topology for general case
        observation, std_observation = general_communication_topology(states, env.minimize)
    return observation, std_observation


def train_policy(config:dict, envs: OptEnvCache, policy: MAPPO):
    """
    Train the policy.
    :param config: The configuration.
    :param envs: The environment.
    :param policy: The policy.
    """
    max_episode = config['environment']['max_episode']
    episode_length = config['episode_length']
    env_id = 0
    for episode in max_episode:
        env_name = config['environment_config']['env_list'][env_id] +


