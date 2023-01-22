"""
DeepHive Algorithm Implementation: A multi-agent reinforcement learning algorithm for black-box optimization.

"""
#imports
import numpy as np
import random
import math
from typing import Callable, List, Tuple
from opt_env import OptEnv, OptEnvCache
from mappo import MAPPO
import os
from argparse import ArgumentParser
import yaml
from optimization_functions import *
import imageio
import re
from log import log_param
import matplotlib.pyplot as plt
from datetime import datetime
from utils import generate_observations, animate, prepare_environment, initialize_policy
import neptune.new as neptune
from decouple import config as conf
from distutils.util import strtobool

def get_args():
    """
    Get the command line arguments.
    :return: The command line arguments.
    """
    # create the parser
    parser = ArgumentParser()
    # add the arguments
    parser.add_argument('-c', '--config', type=str, default='config.yaml', help='The path to the config file.')
    parser.add_argument('-m', '--mode', type=str, default='train', help='The mode of the algorithm. (train, test, plot)')
    parser.add_argument('-e', '--env', type=str, default=0, help='The environment to train/test the algorithm on.')
    parser.add_argument('-l', '--log', type=str, default='False', help='Use neptune for logging.')
    parser.add_argument('-t', '--tags', type=str, default='None', help='Tags for neptune logging.')
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

def train_policy(config:dict, envs: OptEnvCache, policy: MAPPO, neptune_flag: bool, neptune_key: str, neptune_tags: str = None):
    """
    Train the policy.
    :param config: The configuration.
    :param envs: The environment.
    :param policy: The policy.
    """
    max_episode = config['environment_config']['max_episode']
    env_ids = config['environment_config']['env_id']
    decay_rate = config['policy_config']['std_decay_rate']
    log_directory = config['environment_config']['log_directory'] + datetime.now().strftime("%Y%m%d-%H%M%S") + "/" 
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    model_dir = config['policy_config']['model_path'] + datetime.now().strftime("%Y%m%d-%H%M%S") + "_model/" 
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    gif_dir = config['environment_config']['gif_directory'] + "train/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    reward_log_freq = config['policy_config']['reward_log_freq']
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    # create model directory if not present
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if neptune_flag == True:
        run = neptune.init(
            project="elo/deephiveII",
            tags=["train", neptune_tags],
            api_token=neptune_key,
            source_files=["*.py", "config/train_config.yaml", "requirements.txt", "optimization_functions.py", "log.py", "utils.py", "mappo.py", "opt_env.py", "deephive.py"])

        run["parameters"] = config

    for episode in range(max_episode):
        if episode % int(config['environment_config']['change_freq']) == 0:
            env_id = random.choice(env_ids)
            print(f"Episode {episode} of {max_episode} ({episode/max_episode*100:.2f}% ) for {config['environment_config']['env_list'][env_id]} environment")
        env_name = config['environment_config']['env_list'][env_id] + "_" + f"{config['environment_config']['n_dim']}D_{config['environment_config']['n_agents']}_agents"
        env = envs.get_env(env_name)
        _ = env.reset()
        for step in range(env.ep_length):
            # get observations and std_obs
            obs, std_obs = generate_observations(env, split=config['policy_config']['split_agent'], std_obs_type=config['policy_config']['std_obs_type'], )
            # get actions
            agent_actions = []
            for dim in range(env.n_dim):
                action = policy.select_action(obs[dim], std_obs[dim], env.refinement_idx)
                agent_actions.append(action)

            actions = np.transpose(np.array(agent_actions))
            # print("obs", obs)
            # print("actions", actions)
            # step the environment
            next_states, rewards, dones, _ = env.step(actions)
            ## add reward to the buffer
            for agent in range(env.n_agents):
                if policy.split_agent:
                    if agent in env.refinement_idx:
                        policy.exploitation_buffer.rewards += [rewards[agent]]*env.n_dim
                        policy.exploitation_buffer.is_terminals += [dones[agent]]*env.n_dim
                        policy.exploration_buffer.rewards += [rewards[agent]]*env.n_dim
                        policy.exploration_buffer.is_terminals += [dones[agent]]*env.n_dim
                else:
                    policy.buffer.rewards += [rewards[agent]] * env.n_dim
                    policy.buffer.is_terminals += [dones[agent]] * env.n_dim
                # if episode % reward_log_freq:
                #     if neptune_flag == True:
                #         run[f"train/episode_reward/agent-{agent}"].log(rewards[agent])
            if episode % config['environment_config']['log_interval'] == 0: 
                if env.best_agent_value is not None:
                    log_param(f"{episode}-best_agent_value", env.best_agent_value, episode, log_directory + "train_best.json")
                    if neptune_flag == True:
                        run[f"train/episode_run/episode-{episode}-best_agent_value"].log(env.best_agent_value)
            # stop if there is more than half done in dones
            if np.sum(dones) > env.n_agents/2:
                print(f"More than {np.sum(dones)} agents are done in episode {episode} at step {step}")
                break
    
        exploit_return = np.array(env.episode_return["exploit_agents"])
        explore_return = np.array(env.episode_return["explore_agents"])
        all_return = np.array(env.episode_return["all_agents"])
        #print(f"Exploit return: {exploit_return}\n Explore return: {explore_return}")
        episode_return = {"exploit_return": np.mean(exploit_return), "explore_return": np.mean(explore_return), "all_return": np.mean(all_return)}
        # name the log file with the date and time
        if episode % config['policy_config']['reward_log_freq'] == 0: 
            episode_info = {"episode": episode, "env_id": env_id, "env_name": config['environment_config']['env_list'][env_id]}
            log_param(f"episode_return", episode_return , episode_info, log_directory + "train_return.json")
            if neptune_flag == True:
                if policy.split_agent:
                    run[f"train/episode_return/exploit_return"].log(np.mean(exploit_return))
                    run[f"train/episode_return/explore_return"].log(np.mean(explore_return))
                run[f"train/episode_return/all_return"].log(np.mean(all_return))
            # log best agent value from the environment
            

        # update the policy
        if episode % config['policy_config']['update_interval'] == 0:
            policy.update()
            #print(f"Updated the policy at episode {episode} for {config['environment_config']['env_list'][env_id]} environment")
        # decay policy std
        if episode % config['policy_config']['std_decay_freq'] == 0:
            if policy.split_agent:
                policy.exploitation_policy.std.decay_fixed_std(decay_rate)
                if episode % config['policy_config']['std_decay_freq']*2 == 0:
                    policy.exploration_policy.std.decay_fixed_std(decay_rate)
                print(f"Decayed the policy std at episode {episode} for {config['environment_config']['env_list'][env_id]} environment to {policy.exploitation_policy.std.init_std} and {policy.exploration_policy.std.init_std}")
            else:
                policy.policy.std.decay_fixed_std(decay_rate)
                print(f"Decayed the policy std at episode {episode} for {config['environment_config']['env_list'][env_id]} environment to {policy.policy.std.init_std}")
           
        # save the model
        if episode % config['policy_config']['save_interval'] == 0:
            model_path = model_dir + f"_{episode}"
            policy.save(model_path)
            # log model to neptune
            if neptune_flag == True:
                if policy.split_agent:
                    run[f"train/episode_model/exploit_model"].upload(model_path + "_exploitation.pth")
                    run[f"train/episode_model/explore_model"].upload(model_path + "_exploration.pth")
                else:
                    run[f"train/episode_model/episode-{episode}"].upload(model_path + ".pth")
        # animate the training process
        if episode % config['policy_config']['animate_interval'] == 0:
            function_name = config['environment_config']['env_list'][env_id] + str(episode)
            animate(env, config, gif_dir, function_name)
            # log gif to neptune
            if episode % config['environment_config']['log_interval'] == 0:
                if neptune_flag == True:
                    run[f"train/episode_gif/episode-{episode}"].upload(gif_dir + f"{function_name}.gif")
            

def test_policy(config:dict, envs: OptEnvCache, policy: MAPPO, env_id: int):
    """
    Test the policy.
    :param config: The configuration.
    :param envs: The environment.
    :param policy: The policy.
    """

    env_name = config['environment_config']['env_list'][env_id] + "_" + f"{config['environment_config']['n_dim']}D_{config['environment_config']['n_agents']}_agents"
    env = envs.get_env(env_name)
    _ = env.reset()
    model_name = config['policy_config']['test_model_path']
    gif_dir = config['environment_config']['gif_directory'] + "test/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    # load model
    iters = "test"
    policy.load(model_name)
    if policy.split_agent:
        policy.exploitation_policy.std.fixed_std = False
        policy.exploration_policy.std.fixed_std = False
    else:
        policy.policy.fixed_std = False
    for step in range(env.ep_length):
        # get observations and std_obs
        obs, std_obs = generate_observations(env, split=config['policy_config']['split_agent'], std_obs_type=config['test_policy_config']['std_obs_type'])
        # get actions
        agent_actions = []
        for dim in range(env.n_dim):
            action = policy.select_action(obs[dim], std_obs[dim], env.refinement_idx)
            agent_actions.append(action)
        # print(env.refinement_idx)
        actions = np.transpose(np.array(agent_actions))
        #print(actions)
        #print(policy.exploration_policy.std.init_std)
        # step the environment
        next_states, rewards, dones, _ = env.step(actions)
        # stop if there is more than half done in dones
        if np.sum(dones) > env.n_agents/3:
            print(f"More than {np.sum(dones)} agents are done at step {step}")
            break
    function_name = config['environment_config']['env_list'][env_id] + "_" + str(iters)
    animate(env, config, gif_dir, function_name, low_quality=False)
    print(env.get_best_agent())

def main():
    # load the configuration
    args = get_args()
    config_path = args.config
    mode = args.mode
    env_id = int(args.env)
    tags = args.tags
    neptune_flag = bool(strtobool(str(args.log)))
    config = get_config(config_path)
    neptune_key = conf("neptune_secret_access_key", default='')
    # create the environment
    envs = prepare_environment(config, load_from_file=True)
    # create the policy
    
    # train the policy
    if mode == "train": 
        policy = initialize_policy(config['policy_config'])
        train_policy(config, envs, policy, neptune_flag=neptune_flag, neptune_key=neptune_key, neptune_tags=tags)
    # test the policy
    elif mode == "test":
        policy = initialize_policy(config['test_policy_config'])
        test_policy(config, envs, policy, env_id)
    else:
        raise ValueError("Unknown mode")

if __name__ == "__main__":
    main()