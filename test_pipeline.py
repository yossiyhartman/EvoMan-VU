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

import random
random.seed(10)
np.random.seed(10)


def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

def initialize_population(population_size, lower, upper, n_weights = 265):
    return np.random.uniform(lower, upper, (population_size, n_weights))

def parent_selection(pop, pop_fit, n_parents, smoothing = 1):
    fitness  = pop_fit + smoothing - np.min(pop_fit)

    # Fitness proportional selection probability
    fps = fitness / np.sum (fitness)

    # make a random selection of indices
    parent_indices = np.random.choice (np.arange(0,pop.shape[0]), (n_parents,2), p=fps)
    return pop [parent_indices]

def crossover(parents):
    parentsA, parentsB = np.hsplit (parents,2)
    roll = np.random.uniform (size = parentsA.shape)
    offspring = parentsA * (roll >= 0.5) + parentsB * (roll < 0.5)
    # squeeze to get rid of the extra dimension created during parent selecting
    return np.squeeze(offspring,1)

def survivor_selection(pop, pop_fit, n_pop):
    '''
    Replace this function with the class and adjust the code properly
    '''
    best_fit_indices = np.argsort(pop_fit * -1) # -1 since we are maximizing
    survivor_indices = best_fit_indices [:n_pop] # round robbing 
    return pop [survivor_indices], pop_fit[survivor_indices]

def mutate(pop,min_value,max_value, sigma):
    mutation = np.random.normal(0, sigma, size=pop.shape)
    new_pop = pop + mutation
    new_pop[new_pop>max_value] = max_value
    new_pop[new_pop<min_value] = min_value
    return new_pop

experiment_name = 'test1'

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes environment for single objective mode (specialist)  with static enemy and ai player
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

for i in range (generations):
    parents=parent_selection(pop, pop_fitness, len(pop_fitness), smoothing = 1)
    offspring = crossover (parents)
    offspring = mutate (offspring, weight_lower_bound, weight_upper_bound, mutation_sigma)
    offspring_fit = evaluate(env, offspring)
    pop = np.vstack((pop,offspring))
    pop_fit = np.concatenate([pop_fitness,offspring_fit])
    pop, pop_fit = survivor_selection(pop, pop_fit, population_size)
    print (f"Gen {i} - Best: {np.max (pop_fit)} - Mean: {np.mean(pop_fit)}")

fittest_index = np.argmax (pop_fit)
fittest_individual = pop[fittest_index]


#Play with winner

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
	
	#Update the enemy
	env.update_parameter('enemies',[en])

	env.play(sol)

print('\n  \n')