import os
import random
import numpy as np
from evoman.environment import Environment
from demo_controller import player_controller

from operators.parent_selection_methods import ParentSelectionMethods
from operators.recombination_methods import RecombinationMethods
from operators.mutation_methods import MutationMethods
from operators.selection_methods import SelectionMethods

from utils.helpers import *

random.seed(10)
np.random.seed(10)

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
    parents = ParentSelectionMethods().select_parents(pop, pop_fitness, len(pop_fitness), smoothing=1, method="fitness_prop_selection")

    unmutated_offspring = RecombinationMethods().create_offspring(parents, method="one_point_crossover")
    
    offspring = MutationMethods().apply_mutation(unmutated_offspring, weight_lower_bound, weight_upper_bound, mutation_sigma)
    offspring_fit = evaluate(env, offspring)
    
    pop = np.vstack((pop, offspring))
    pop_fit = np.concatenate([pop_fitness, offspring_fit])
    pop, pop_fit = SelectionMethods().select_survivor(pop, pop_fit, population_size, method="elitism")
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
