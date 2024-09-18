import numpy as np

def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env, y), x)))

def initialize_population(population_size, lower, upper, n_weights=265):
    return np.random.uniform(lower, upper, (population_size, n_weights))

# Deze functies moeten we verplaatsen naar de operators folder
def parent_selection(pop, pop_fit, n_parents, smoothing=1):
    fitness = pop_fit + smoothing - np.min(pop_fit)

    # Fitness proportional selection probability
    fps = fitness / np.sum(fitness)

    # make a random selection of indices
    parent_indices = np.random.choice(np.arange(0, pop.shape[0]), (n_parents, 2), p=fps)
    return pop[parent_indices]

def crossover(parents):
    parentsA, parentsB = np.hsplit(parents, 2)
    roll = np.random.uniform(size=parentsA.shape)
    offspring = parentsA * (roll >= 0.5) + parentsB * (roll < 0.5)
    # squeeze to get rid of the extra dimension created during parent selecting
    return np.squeeze(offspring, 1)

def mutate(pop, min_value, max_value, sigma):
    mutation = np.random.normal(0, sigma, size=pop.shape)
    new_pop = pop + mutation
    new_pop[new_pop > max_value] = max_value
    new_pop[new_pop < min_value] = min_value
    return new_pop