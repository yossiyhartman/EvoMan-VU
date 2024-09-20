import numpy as np

def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env, y), x)))

def initialize_population(population_size, lower, upper, n_weights=265):
    return np.random.uniform(lower, upper, (population_size, n_weights))