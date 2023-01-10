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
import matplotlib.pyplot as plt

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
        obj_func, bounds, opt_obj_value, type = get_obj_func(env_config['env_list'][i])
        if type == 'minimize':
            min = True
        else:
            min = False
        # if n_dim is not 2D, ignore the bounds and use bounds from the config file
        if env_config['n_dim'] != 2:
            bounds = env_config['bounds'][i]
        env = OptEnv(func_name, obj_func, env_config['n_agents'], env_config['n_dim'], 
                    bounds, env_config['ep_length'], min, env_config['freeze'],opt_value=float(opt_obj_value)
                    )
        # cache the environment
        envs_cache.add_env(func_name, env)
    # return the environment
    return envs_cache

def initialize_policy(config:dict) -> MAPPO:
    """
    Initialize the policy.
    :param config: The config.
    :return: The policy.
    """
    # get the policy config
    policy_config = config['policy_config']
    # create the policy
    policy = MAPPO(policy_config['state_dim'], policy_config['action_dim'],config['environment_config']['ep_length'],policy_config['init_std'],
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


def animate(env:OptEnv, env_id, iter, config:dict):
    plot_directory = config['environment_config']['plot_directory']
    env_name = env_name = config['environment_config']['env_list'][env_id] + "_" + f"{config['environment_config']['n_dim']}D_{config['environment_config']['n_agents']}_agents"
    title = env_name + f"_{iter}_.gif"
    opt_func_name = config['environment_config']['env_list'][env_id]
    ep_length = env.ep_length
    fps = config['environment_config']['fps']
    count = 0
    agents_pos = env.stateHistory
    markers = ['o','v','s','p','P','*','h','H','+','x','X','D','d','|','_']
    while True:
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot()
        fig.subplots_adjust(top=0.85)
        plt.clf()
        X =  Y = np.linspace(env.min_pos, env.max_pos, 101)
        x, y = np.meshgrid(X, Y)
        Z = env.optFunc(np.array([x, y]),minimize=env.minimize, plotable=True)
        plt.contour(x, y, Z, 20)
        plt.colorbar()
        for i in range(env.n_agents):
            pos = agents_pos[count][i]
            if i == env.best_agent_idx:
                plt.plot(pos[0], pos[1] ,marker=markers[1], markersize=19, markerfacecolor='b')
            elif i in env.refinement_idx:
                plt.plot(pos[0], pos[1] ,marker=markers[2], markersize=17, markerfacecolor='g')
            else:
                plt.plot(pos[0], pos[1] ,marker=markers[0], markersize=15, markerfacecolor='r')
            plt.text(env.max_pos[0], env.max_pos[0], f"Step {count+1}", style='italic',
            bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 10})
            # add legend
            # set the legend such that the best agent is always the first one, the refinement agents are the second ones and the rest are the exploration agents
            Line2D = plt.Line2D
            legend_elements = [Line2D([0], [0], marker=markers[1], color='w', label='Best Agent', markerfacecolor='b', markersize=15),
                               Line2D([0], [0], marker=markers[2], color='w', label='Refinement Agents', markerfacecolor='g', markersize=15),
                               Line2D([0], [0], marker=markers[0], color='w', label='Exploration Agents', markerfacecolor='r', markersize=15)]
            plt.legend(handles=legend_elements, loc='upper right')
        #plt.pause(0.5)
        plt.title(opt_func_name)
        plt_dir = plot_directory + f"{count}.png"  
        #print(plt_dir)
        plt.savefig(plt_dir) 
        plt.close(fig)
        plt.show()
        count += 1
        if count >= ep_length:
            break
    images = []
    filenames = os.listdir(plot_directory)
    filenames.sort(key=lambda f: int(re.sub('\D', '', f)))
    for filename in filenames:
        images.append(imageio.imread(plot_directory + filename))
    imageio.mimsave(title, images, fps=fps)
    for filename in set(filenames):
        os.remove(plot_directory + filename)


def train_policy(config:dict, envs: OptEnvCache, policy: MAPPO):
    """
    Train the policy.
    :param config: The configuration.
    :param envs: The environment.
    :param policy: The policy.
    """
    max_episode = config['environment_config']['max_episode']
    env_ids = config['environment_config']['env_id']
    # chose random id in list
    for episode in range(max_episode):
        if episode % int(config['environment_config']['change_freq']) == 0:
            env_id = random.choice(env_ids)
            print(f"Episode {episode} of {max_episode} ({episode/max_episode*100:.2f}% ) for {config['environment_config']['env_list'][env_id]} environment")
        env_name = config['environment_config']['env_list'][env_id] + "_" + f"{config['environment_config']['n_dim']}D_{config['environment_config']['n_agents']}_agents"
        env = envs.get_env(env_name)
        _ = env.reset()
        for step in range(env.ep_length):
            # get observations and std_obs
            obs, std_obs = generate_observations(env, split=config['policy_config']['split_agent'])
            # get actions
            agent_actions = []
            for dim in range(env.n_dim):
                action = policy.select_action(obs[dim], std_obs[dim], step, env.refinement_idx)
                agent_actions.append(action)

            actions = np.transpose(np.array(agent_actions))
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
            # stop if there is more than half done in dones
            if np.sum(dones) > env.n_agents/2:
                print(f"More than {np.sum(dones)} agents are done in episode {episode} at step {step}")
                break
                
        # update the policy
        if episode % config['policy_config']['update_interval'] == 0:
            policy.update()
            print(f"Updated the policy at episode {episode} for {config['environment_config']['env_list'][env_id]} environment")
        # save the model
        if episode % config['policy_config']['save_interval'] == 0:
            model_path = "models/" + config['policy_config']['model_path'] + f"_{episode}"
            policy.save(model_path)
        # animate the training process
        if episode % config['policy_config']['animate_interval'] == 0:
            animate(env, env_id, episode, config)
        
            

def test_policy(config:dict, envs: OptEnvCache, policy: MAPPO, env_id: int):
    """
    Test the policy.
    :param config: The configuration.
    :param envs: The environment.
    :param policy: The policy.
    """

    env_name = config['environment_config']['env_list'][env_id] + "_" + f"{config['environment_config']['n_dim']}D_{config['environment_config']['n_agents']}_agents"
    env = envs.get_env(env_name)
    states = env.reset()
    model_name = config['policy_config']['test_model_path']
    # load model
    iters = "test"
    policy.load(model_name)
    for step in range(env.ep_length):
        # get observations and std_obs
        obs, std_obs = generate_observations(env, split=config['policy_config']['split_agent'])
        # get actions
        agent_actions = []
        for dim in range(env.n_dim):
            action = policy.select_action(obs[dim], std_obs[dim], step, env.refinement_idx)
            agent_actions.append(action)
        # print(env.refinement_idx)
        actions = np.transpose(np.array(agent_actions))
        # step the environment
        next_states, rewards, dones, _ = env.step(actions)
        # stop if there is more than half done in dones
        if np.sum(dones) > env.n_agents/3:
            print(f"More than {np.sum(dones)} agents are done at step {step}")
            break
        
    animate(env, env_id, iters, config)
    print(env.get_best_agent())

def main():
    # load the configuration
    args = get_args()
    config_path = args.config
    mode = args.mode
    env_id = int(args.env)
    config = get_config(config_path)
    # create the environment
    envs = prepare_environment(config)
    # create the policy
    policy = initialize_policy(config)
    # train the policy
    if mode == "train": 
        train_policy(config, envs, policy)
    # test the policy
    elif mode == "test":
        test_policy(config, envs, policy, env_id)
    else:
        raise ValueError("Unknown mode")

if __name__ == "__main__":
    main()