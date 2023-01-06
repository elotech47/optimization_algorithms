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

def main():
    pass

    