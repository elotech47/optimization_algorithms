""" A python implementation of the Particle Swarm Optimization algorithm."""

# imports
import numpy as np
import random
import math
from typing import Callable, List, Tuple

class PSO():
    """ A python implementation of the Particle Swarm Optimization algorithm."""
    def __init__(self, param_size:int, num_particles:int, num_iterations:int, fitness_function:Callable, bounds:list, inertia=0.5, cognitive=1.0, social=1.0, min=False) -> None:
        """ Initialize the PSO class.
        :param param_size: The number of parameters in the fitness function.
        :param num_particles: The number of particles in the swarm.
        :param num_iterations: The number of iterations to run the algorithm.
        :param fitness_function: The fitness function to be optimized.
        :param bounds: The bounds for each parameter in the fitness function.
        :param inertia: The inertia weight.
        :param cognitive: The cognitive weight.
        :param social: The social weight.
        :param min: Whether to minimize or maximize the fitness function.
        """
        self.param_size = param_size
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.fitness_function = fitness_function
        self.bounds = bounds
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.min = min
    
    def optimize(self) -> Tuple[float, List[float]]:
        """ Optimize the fitness function.
        :return: The best fitness and parameters.
        """
        # initialize the swarm
        swarm = self._initialize_swarm()
        # initialize the best fitness and parameters
        best_fitness = math.inf if self.min else -math.inf
        best_params = []
        # iterate through the number of iterations
        for _ in range(self.num_iterations):
            # iterate through each particle in the swarm
            for particle in swarm:
                # calculate the fitness
                fitness = self.fitness_function(particle)
                # check if the fitness is better than the best fitness
                if self.min:
                    if fitness < best_fitness:
                        best_fitness = fitness
                        best_params = particle
                else:
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_params = particle
            # update the swarm
            swarm = self._update_swarm(swarm)
        return best_fitness, best_params
    
    def _initialize_swarm(self) -> List[List[float]]:
        """ Initialize the swarm.
        :return: The swarm.
        """
        swarm = []
        for _ in range(self.num_particles):
            particle = []
            for i in range(self.param_size):
                # get the lower and upper bounds
                lower_bound = self.bounds[i][0]
                upper_bound = self.bounds[i][1]
                # generate a random value within the bounds
                particle.append(random.uniform(lower_bound, upper_bound))
            swarm.append(particle)
        return swarm

    def _update_swarm(self, swarm:List[List[float]]) -> List[List[float]]:
        """ Update the swarm.
        :param swarm: The swarm.
        :return: The updated swarm.
        """
        # get the best particle in the swarm
        best_particle = self._get_best_particle(swarm)
        # iterate through each particle in the swarm
        for particle in swarm:
            # update the particle
            particle = self._update_particle(particle, best_particle)
        return swarm

    def _get_best_particle(self, swarm:List[List[float]]) -> List[float]:
        """ Get the best particle in the swarm.
        :param swarm: The swarm.
        :return: The best particle.
        """
        best_particle = []
        best_fitness = math.inf
        # iterate through each particle in the swarm
        for particle in swarm:
            # calculate the fitness
            fitness = self.fitness_function(particle)
            # check if the fitness is better than the best fitness
            if self.min:
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_particle = particle
            else:
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_particle = particle
        return best_particle