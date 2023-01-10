""" A utility file containing functions for optimizing a fitness function. """
import math
from typing import Callable, List, Tuple
import numpy as np

# create the sphere function that takes an array of parameters and returns an array of fitness values
def sphere(params:np.ndarray, minimize=False, plotable=False) -> np.ndarray:
    """ The sphere function.
    :param params: The parameters.
    :param minimize: Whether to minimize or maximize the fitness function.
    :param plotable: Whether the parameters are plotable.
    :return: The fitness.
    """
    # check the input is right
    if not isinstance(params, np.ndarray):
        raise TypeError('The parameters must be a numpy array.')
    # calculate the fitness
    # check if params is a 1D array and reshape it if it is
    if len(params.shape) == 1:
        params = params.reshape(1, -1)
    if plotable:
        # check if param is not 2D and raise an error if it is
        if params.shape[0] != 2:
            raise ValueError('The parameters must be a 2D numpy array.')
        param1 = params[0, :]
        param2 = params[1, :]
        fitness = np.square(param1) + np.square(param2)
    else:
        fitness = np.sum(np.square(params), axis=1)
    # check if the fitness should be minimized
    return fitness if minimize else -fitness


# create the rosenbrock function that takes an array of parameters and returns an array of fitness values
def rosenbrock(params:np.ndarray, minimize=False, plotable=False) -> np.ndarray:
    """ The rosenbrock function.
    :param params: The parameters.
    :param minimize: Whether to minimize or maximize the fitness function.
    :param plotable: Whether the parameters are plotable.
    :return: The fitness.
    """
    # check the input is right
    if not isinstance(params, np.ndarray):
        raise TypeError('The parameters must be a numpy array.')
    # calculate the fitness
    # check if params is a 1D array and reshape it if it is
    if len(params.shape) == 1:
        params = params.reshape(1, -1)
    if plotable:
        # check if param is not 2D and raise an error if it is
        if params.shape[0] != 2:
            raise ValueError('The parameters must be a 2D numpy array.')
        param1 = params[0, :]
        param2 = params[1, :]
        fitness = 100 * np.square(param2 - np.square(param1)) + np.square(param1 - 1)
    else:
        fitness = np.sum(100.0 * np.square(params[:, 1:] - np.square(params[:, :-1])) + np.square(params[:, :-1] - 1.0), axis=1)
    # check if the fitness should be minimized
    return fitness if minimize else -fitness

# # cosine mixture function that takes an array of parameters and returns an array of fitness values
def cosine_mixture(params:np.ndarray, minimize=False, plotable=False) -> np.ndarray:
    """ The cosine mixture function.
    :param params: The parameters.
    :param minimize: Whether to minimize or maximize the fitness function.
    :param plotable: Whether the parameters are plotable.
    :return: The fitness.
    """
    # check the input is right
    if not isinstance(params, np.ndarray):
        raise TypeError('The parameters must be a numpy array.')
    # calculate the fitness
    # check if params is a 1D array and reshape it if it is
    if len(params.shape) == 1:
        params = params.reshape(1, -1)
    if plotable:
        # check if param is not 2D and raise an error if it is
        if params.shape[0] != 2:
            raise ValueError('The parameters must be a 2D numpy array.')
        param1 = params[0, :]
        param2 = params[1, :]
        fitness = 0.1 * (np.cos(5 * math.pi * param1) + np.cos(5 * math.pi * param2)) - (np.square(param1) + np.square(param2))
    else:
        fitness = 0.1 * np.sum(np.cos(5 * math.pi * params), axis=1) - np.sum(np.square(params), axis=1)
    # check if the fitness should be minimized
    return -fitness if minimize else fitness

def ghabit_function_2D(X):
    x1 = X[0,:]
    x0 = X[1,:]
    y = (1 - x1/2 + x1**5 + x0**3) * np.exp(-x1**2 - x0**2)
    return -y

# ghabit function that takes an array of parameters and returns an array of fitness values
def ghabit(params:np.ndarray, minimize=False, plotable=False) -> np.ndarray:
    """ The ghabit function.
    :param params: The parameters.
    :param minimize: Whether to minimize or maximize the fitness function.
    :param plotable: Whether the parameters are plotable.
    :return: The fitness.
    """
    # check the input is right
    if not isinstance(params, np.ndarray):
        raise TypeError('The parameters must be a numpy array.')
    # calculate the fitness
    # check if params is a 1D array and reshape it if it is
    if len(params.shape) == 1:
        params = params.reshape(1, -1)
    if plotable:
        # check if param is not 2D and raise an error if it is
        if params.shape[0] != 2:
            raise ValueError('The parameters must be a 2D numpy array.')
        param1 = params[0, :]
        param2 = params[1, :]
        fitness = (1 - param1 / 2 + param1 ** 5 + param2 ** 3) * np.exp(-param1 ** 2 - param2 ** 2)
    else:
        fitness = (1 - params[:, 0] / 2 + params[:, 0] ** 5 + params[:, 1] ** 3) * np.exp(-params[:, 0] ** 2 - params[:, 1] ** 2)
    # check if the fitness should be minimized
    return fitness if minimize else -fitness

def cos_function_2D(X):
    x1 = X[0,:]
    x0 = X[1,:]
    y = (np.cos(x1-2) + np.cos(x0-2)) + (np.cos(2*x1-4) + np.cos(2*x0-4)) + (np.cos(4*x1-8) + np.cos(4*x0-8))
    return -y

# use the formula in cos_function_2D to create a fitness function
def cos_function(params:np.ndarray, minimize=False, plotable=False) -> np.ndarray:
    """ The cos function.
    :param params: The parameters.
    :param minimize: Whether to minimize or maximize the fitness function.
    :param plotable: Whether the parameters are plotable.
    :return: The fitness.
    """
    # check the input is right
    if not isinstance(params, np.ndarray):
        raise TypeError('The parameters must be a numpy array.')
    # calculate the fitness
    # check if params is a 1D array and reshape it if it is
    if len(params.shape) == 1:
        params = params.reshape(1, -1)
    if plotable:
        # check if param is not 2D and raise an error if it is
        if params.shape[0] != 2:
            raise ValueError('The parameters must be a 2D numpy array.')
        param1 = params[0, :]
        param2 = params[1, :]
        fitness = np.cos(param2 - 2) + np.cos(param1 - 2) + (np.cos(2 * param2 - 4) + np.cos(2 * param1 - 4)) + (np.cos(4 * param2 - 8) + np.cos(4 * param1 - 8))
    else:
        fitness = np.sum(np.cos(params - 2), axis=1) + (np.sum(np.cos(2 * params - 4), axis=1)) + (np.sum(np.cos(4 * params - 8), axis=1))
    # check if the fitness should be minimized
    return -fitness if minimize else fitness

# Griewank function that takes an array of parameters and returns an array of fitness values
def griewank(params:np.ndarray, minimize=False, plotable=False) -> np.ndarray:
    """ The griewank function.
    :param params: The parameters.
    :param minimize: Whether to minimize or maximize the fitness function.
    :param plotable: Whether the parameters are plotable.
    :return: The fitness.
    """
    # check the input is right
    if not isinstance(params, np.ndarray):
        raise TypeError('The parameters must be a numpy array.')
    # calculate the fitness
    # check if params is a 1D array and reshape it if it is
    if len(params.shape) == 1:
        params = params.reshape(1, -1)
    if plotable:
        # check if param is not 2D and raise an error if it is
        if params.shape[0] != 2:
            raise ValueError('The parameters must be a 2D numpy array.')
        param1 = params[0, :]
        param2 = params[1, :]
        fitness = 1 + np.square(param1) / 4000 + np.square(param2) / 4000 - np.cos(param1) * np.cos(param2 / np.sqrt(2))
    else:
        fitness = 1 + np.sum(np.square(params), axis=1) / 4000 - np.prod(np.cos(params / np.sqrt(np.arange(1, params.shape[1] + 1))), axis=1)
    # check if the fitness should be minimized
    return fitness if minimize else -fitness

# Rastrigin function that takes an array of parameters and returns an array of fitness values
def rastrigin(params:np.ndarray, minimize=False, plotable=False) -> np.ndarray:
    """ The rastrigin function.
    :param params: The parameters.
    :param minimize: Whether to minimize or maximize the fitness function.
    :param plotable: Whether the parameters are plotable.
    :return: The fitness.
    """
    # check the input is right
    if not isinstance(params, np.ndarray):
        raise TypeError('The parameters must be a numpy array.')
    # calculate the fitness
    # check if params is a 1D array and reshape it if it is
    if len(params.shape) == 1:
        params = params.reshape(1, -1)
    if plotable:
        # check if param is not 2D and raise an error if it is
        if params.shape[0] != 2:
            raise ValueError('The parameters must be a 2D numpy array.')
        param1 = params[0, :]
        param2 = params[1, :]
        fitness = 10 * 2 + np.square(param1) - 10 * np.cos(2 * math.pi * param1) + np.square(param2) - 10 * np.cos(2 * math.pi * param2)
    else:
        fitness = 10 * params.shape[1] + np.sum(np.square(params), axis=1) - 10 * np.sum(np.cos(2 * math.pi * params), axis=1)
    # check if the fitness should be minimized
    return fitness if minimize else -fitness

# three hump camel function that takes an array of parameters and returns an array of fitness values
def three_hump_camel(params:np.ndarray, minimize=False, plotable=False) -> np.ndarray:
    """ The three hump camel function.
    :param params: The parameters.
    :param minimize: Whether to minimize or maximize the fitness function.
    :param plotable: Whether the parameters are plotable.
    :return: The fitness.
    """
    # check the input is right
    if not isinstance(params, np.ndarray):
        raise TypeError('The parameters must be a numpy array.')
    # calculate the fitness
    # check if params is a 1D array and reshape it if it is
    if len(params.shape) == 1:
        params = params.reshape(1, -1)
    if plotable:
        # check if param is not 2D and raise an error if it is
        if params.shape[0] != 2:
            raise ValueError('The parameters must be a 2D numpy array.')
        param1 = params[0, :]
        param2 = params[1, :]
        fitness = 2 * np.square(param1) - 1.05 * np.power(param1, 4) + np.power(param1, 6) / 6 + param1 * param2 + np.square(param2)
    else:
        fitness = 2 * np.square(params[:, 0]) - 1.05 * np.power(params[:, 0], 4) + np.power(params[:, 0], 6) / 6 + params[:, 0] * params[:, 1] + np.square(params[:, 1])
    # check if the fitness should be minimized
    return fitness if minimize else -fitness

# six hump camel function that takes an array of parameters and returns an array of fitness values
def six_hump_camel(params:np.ndarray, minimize=False, plotable=False) -> np.ndarray:
    """ The six hump camel function.
    :param params: The parameters.
    :param minimize: Whether to minimize or maximize the fitness function.
    :param plotable: Whether the parameters are plotable.
    :return: The fitness.
    """
    # check the input is right
    if not isinstance(params, np.ndarray):
        raise TypeError('The parameters must be a numpy array.')
    # calculate the fitness
    # check if params is a 1D array and reshape it if it is
    if len(params.shape) == 1:
        params = params.reshape(1, -1)
    if plotable:
        # check if param is not 2D and raise an error if it is
        if params.shape[0] != 2:
            raise ValueError('The parameters must be a 2D numpy array.')
        param1 = params[0, :]
        param2 = params[1, :]
        fitness = (4 - 2.1 * np.square(param1) + np.power(param1, 4) / 3) * np.square(param1) + param1 * param2 + (-4 + 4 * np.square(param2)) * np.square(param2)
    else:
        fitness = (4 - 2.1 * np.square(params[:, 0]) + np.power(params[:, 0], 4) / 3) * np.square(params[:, 0]) + params[:, 0] * params[:, 1] + (-4 + 4 * np.square(params[:, 1])) * np.square(params[:, 1])
    # check if the fitness should be minimized
    return fitness if minimize else -fitness

# Easom function that takes an array of parameters and returns an array of fitness values
def easom(params:np.ndarray, minimize=False, plotable=False) -> np.ndarray:
    """ The easom function.
    :param params: The parameters.
    :param minimize: Whether to minimize or maximize the fitness function.
    :param plotable: Whether the parameters are plotable.
    :return: The fitness.
    """
    # check the input is right
    if not isinstance(params, np.ndarray):
        raise TypeError('The parameters must be a numpy array.')
    # calculate the fitness
    # check if params is a 1D array and reshape it if it is
    if len(params.shape) == 1:
        params = params.reshape(1, -1)
    if plotable:
        # check if param is not 2D and raise an error if it is
        if params.shape[0] != 2:
            raise ValueError('The parameters must be a 2D numpy array.')
        param1 = params[0, :]
        param2 = params[1, :]
        fitness = -np.cos(param1) * np.cos(param2) * np.exp(-np.square(param1 - math.pi) - np.square(param2 - math.pi))
    else:
        fitness = -np.cos(params[:, 0]) * np.cos(params[:, 1]) * np.exp(-np.square(params[:, 0] - math.pi) - np.square(params[:, 1] - math.pi))
    # check if the fitness should be minimized
    return fitness if minimize else -fitness

# Ackley function that takes an array of parameters and returns an array of fitness values
def ackley(params:np.ndarray, minimize=False, plotable=False) -> np.ndarray:
    """ The ackley function.
    :param params: The parameters.
    :param minimize: Whether to minimize or maximize the fitness function.
    :param plotable: Whether the parameters are plotable.
    :return: The fitness.
    """
    # check the input is right
    if not isinstance(params, np.ndarray):
        raise TypeError('The parameters must be a numpy array.')
    # calculate the fitness
    # check if params is a 1D array and reshape it if it is
    if len(params.shape) == 1:
        params = params.reshape(1, -1)
    if plotable:
        # check if param is not 2D and raise an error if it is
        if params.shape[0] != 2:
            raise ValueError('The parameters must be a 2D numpy array.')
        param1 = params[0, :]
        param2 = params[1, :]
        fitness = -20 * np.exp(-0.2 * np.sqrt(0.5 * (np.square(param1) + np.square(param2)))) - np.exp(0.5 * (np.cos(2 * math.pi * param1) + np.cos(2 * math.pi * param2))) + math.e + 20
    else:
        fitness = -20 * np.exp(-0.2 * np.sqrt(0.5 * (np.square(params[:, 0]) + np.square(params[:, 1])))) - np.exp(0.5 * (np.cos(2 * math.pi * params[:, 0]) + np.cos(2 * math.pi * params[:, 1]))) + math.e + 20
    # check if the fitness should be minimized
    return fitness if minimize else -fitness


# bohachevsky function that takes an array of parameters and returns an array of fitness values
def bohachevsky(params:np.ndarray, minimize=False, plotable=False) -> np.ndarray:
    """ The bohachevsky function.
    :param params: The parameters.
    :param minimize: Whether to minimize or maximize the fitness function.
    :param plotable: Whether the parameters are plotable.
    :return: The fitness.
    """
    # check the input is right
    if not isinstance(params, np.ndarray):
        raise TypeError('The parameters must be a numpy array.')
    # calculate the fitness
    # check if params is a 1D array and reshape it if it is
    if len(params.shape) == 1:
        params = params.reshape(1, -1)
    if plotable:
        # check if param is not 2D and raise an error if it is
        if params.shape[0] != 2:
            raise ValueError('The parameters must be a 2D numpy array.')
        param1 = params[0, :]
        param2 = params[1, :]
        fitness = np.square(param1) + 2 * np.square(param2) - 0.3 * np.cos(3 * math.pi * param1) - 0.4 * np.cos(4 * math.pi * param2) + 0.7
    else:
        fitness = np.square(params[:, 0]) + 2 * np.square(params[:, 1]) - 0.3 * np.cos(3 * math.pi * params[:, 0]) - 0.4 * np.cos(4 * math.pi * params[:, 1]) + 0.7
    # check if the fitness should be minimized
    return fitness if minimize else -fitness

# booth function that takes an array of parameters and returns an array of fitness values
def booth(params:np.ndarray, minimize=False, plotable=False) -> np.ndarray:
    """ The booth function.
    :param params: The parameters.
    :param minimize: Whether to minimize or maximize the fitness function.
    :param plotable: Whether the parameters are plotable.
    :return: The fitness.
    """
    # check the input is right
    if not isinstance(params, np.ndarray):
        raise TypeError('The parameters must be a numpy array.')
    # calculate the fitness
    # check if params is a 1D array and reshape it if it is
    if len(params.shape) == 1:
        params = params.reshape(1, -1)
    if plotable:
        # check if param is not 2D and raise an error if it is
        if params.shape[0] != 2:
            raise ValueError('The parameters must be a 2D numpy array.')
        param1 = params[0, :]
        param2 = params[1, :]
        fitness = np.square(param1 + 2 * param2 - 7) + np.square(2 * param1 + param2 - 5)
    else:
        fitness = np.square(params[:, 0] + 2 * params[:, 1] - 7) + np.square(2 * params[:, 0] + params[:, 1] - 5)
    # check if the fitness should be minimized
    return fitness if minimize else -fitness


def get_obj_func(name):
    """ Returns the objective function with the given name.
    :param name: The name of the objective function.
    :return: The objective function and thier bounds.
    """
    return {
        'sphere': (sphere, [(-3, -3),(3, 3)], 0, "maximize"),
        'rosenbrock': (rosenbrock, [(-2.048, 2.048), (-2.048, 2.048)], 0, "minimize"),
        'cosine_mixture': (cosine_mixture, [(-1, -1), (1, 1)], 0.2, "maximize"),
        'ghabit': (ghabit, [(-3, -3),(3, 3)], 1.058, "maximize"),
        'cos_function': (cos_function, [(-2, -2),(4, 4)], 6, "maximize"),
        'ackley': (ackley, [(-32, -32),(32, 32)], 0, "minimize"),
        'rastrigin': (rastrigin, [(-5.12, -5.12),(5.12, 5.12)], 0, "minimize"),
        'griewank': (griewank, [(-600, -600),(600, 600)], 0, "minimize"),
        'bohachevsky': (bohachevsky , [(-100, -100),(100, 100)], 0, "minimize"),
        'booth': (booth, [(-10, -10),(10, 10)], 0, "minimize"),
        'three_hump_camel': (three_hump_camel, [(-5, -5),(5, 5)], 0, "minimize"),
        'easom': (easom, [(-100, -100),(100, 100)], -1, "minimize"),
        'six_hump_camel': (six_hump_camel, [(-3, -2),(3, 2)], -1.0316, "minimize"),
    }.get(name)
    
