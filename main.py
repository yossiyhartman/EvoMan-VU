# imports framework
import sys

# imports other libs
import numpy as np
import os
import math
import itertools
import pandas as pd
import sys, os
from evoman.environment import Environment
from demo_controller import player_controller
from operators.selection_methods import SelectionMethods
from helpers import *

import random
random.seed(10)
np.random.seed(10)


def survivor_selection(pop, pop_fit, n_pop, method="elitism"):
    """
    Replaces the old survivor_selection function using the SelectionMethods class.
    """
    population_with_fitness = [(pop[i], pop_fit[i]) for i in range(len(pop))]
    selection = SelectionMethods(population_with_fitness)

    # Choose the survivor selection method
    if method == "elitism":
        survivors = selection.elitism(n_pop)
    elif method == "fitness_proportionate":
        survivors = selection.fitness_proportionate_selection(n_pop)
    elif method == "tournament":
        survivors = selection.tournament_selection(tournament_size=3, num_survivors=n_pop)
    elif method == "rank_based":
        survivors = selection.rank_based_selection(n_pop)
    elif method == "steady_state":
        # For steady-state, you would provide offspring as additional input
        offspring = []  # Needs to be passed accordingly
        survivors = selection.steady_state_selection(offspring, num_replacements=10)
    elif method == "truncation":
        survivors = selection.truncation_selection(truncation_threshold=0.5)
    elif method == "random":
        survivors = selection.random_replacement(num_replacements=10)
    else:
        raise ValueError(f"Unknown selection method: {method}")

    # Extract individuals and fitness from survivors
    pop_survivors, pop_fit_survivors = zip(*survivors)
    return np.array(pop_survivors), np.array(pop_fit_survivors)


experiment_name = 'test1'

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes environment for single objective mode (specialist) with static enemy and AI player
env = Environment(experiment_name=experiment_name,
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  speed="fastest",
                  enemymode="static",
                  level=2,
                  visuals=False)

population_size = 100
n_evaluations = 3
n_offspring = 50
weight_upper_bound = 2
weight_lower_bound = -2
mutation_sigma = .3
generations = 30

pop = initialize_population(population_size, -2, 2)
pop_fitness = evaluate(env, pop)

for i in range(generations):
    parents = parent_selection(pop, pop_fitness, len(pop_fitness), smoothing=1)
    offspring = crossover(parents)
    offspring = mutate(offspring, weight_lower_bound, weight_upper_bound, mutation_sigma)
    offspring_fit = evaluate(env, offspring)
    pop = np.vstack((pop, offspring))
    pop_fit = np.concatenate([pop_fitness, offspring_fit])
    
    # Use the new survivor selection function with elitism method
    pop, pop_fit = survivor_selection(pop, pop_fit, population_size, method="elitism")
    
    print(f"Gen {i} - Best: {np.max(pop_fit)} - Mean: {np.mean(pop_fit)}")

fittest_index = np.argmax(pop_fit)
fittest_individual = pop[fittest_index]

# Play with winner
sol = fittest_individual

env = Environment(experiment_name=experiment_name,
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  speed="normal",
                  enemymode="static",
                  level=2,
                  visuals=True)

# tests saved demo solutions for each enemy
enemy_list = [1]
for en in enemy_list:
    # Update the enemy
    env.update_parameter('enemies', [en])

    env.play(sol)

print('\n\n')
