import os
import json
from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np

experiment_name = "Yossi GA"

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(
    experiment_name=experiment_name,
    enemies=[4],
    playermode="ai",
    player_controller=player_controller(n_hidden_neurons),
    enemymode="static",
    level=2,
    speed="fastest",
    visuals=False,
)


def simulation(x):
    f, p, e, t = env.play(pcont=x)
    return f


def evaluate(x):
    return np.array(list(map(lambda y: simulation(y), x)))


def norm(x, pfit_pop):
    if (max(pfit_pop) - min(pfit_pop)) > 0:
        x_norm = (x - min(pfit_pop)) / (max(pfit_pop) - min(pfit_pop))
    else:
        x_norm = 0

    if x_norm <= 0:
        x_norm = 0.0000000001
    return x_norm


class GA:

    def __init__(self, n_genomes, population_size, n_offspring, mutation_p) -> None:
        self.n_genomes = n_genomes
        self.population_size = population_size
        self.n_offspring = n_offspring
        self.mutation_p = mutation_p

    ###################
    # INITIALIZING
    ###################

    def initialize_population(self) -> np.array:
        """
            creates a population with random genomes

        Returns:
            np.array: _description_
        """

        return np.random.normal(size=(self.population_size, self.n_genomes))

    ###################
    # SELECTION
    ###################

    def tournament_selection(self, population: np.array, fitness: np.array) -> np.array:
        """
            Randomly samples two species from a population and returns the individual with the highest fitness score

        Args:
            population (np.array): An array that holds all individuals
            fitness (np.array): fitness values of all individuals

        Returns:
            np.array: the winning individual
        """

        index_p1, index_p2 = np.random.randint(0, self.population_size, 2)
        return population[index_p1] if fitness[index_p1] > fitness[index_p2] else population[index_p2]

    def eletist_selection(self, population: np.array, fitness: np.array, frac: float = 0.1) -> np.array:
        """
            Takes the fraction of the population with the best fitness score

        Args:
            population (np.array): An array that holds all individuals
            fitness (np.array): fitness values of all individuals
            frac (float, optional): The faction of the population. Defaults to .1.

        Returns:
            np.array: the top performing individuals
        """

        population_zip = zip(population, fitness)

        population_sorted = sorted(population_zip, key=lambda x: x[1], reverse=True)

        population_slice = int(np.floor(frac * self.population_size))

        return [pop for pop, _ in population_sorted][0:population_slice]

    ###################
    # MUTATION
    ###################

    def mutate(self, individual: np.array) -> np.array:
        """
            each genome of an individual is randomly mutated

        Args:
            individual (np.array): individual from population
            sigma (float, optional): sigma of normal distribution. Defaults to 1.0.

        Returns:
            np.array: mutated individual
        """

        mutation_dist = np.random.uniform(0, 1, size=self.n_genomes) <= 0.5
        individual += mutation_dist * np.random.normal(0, self.mutation_p, size=self.n_genomes)
        return individual

    ###################
    # CROSSOVER
    ###################

    def crossover(self, population: np.array, fitness: np.array) -> np.array:
        """
            Performs crossover based on tournament selection. Subsequently, applies mutation to random individuals

        Args:
            population (np.array): An array that holds all individuals
            fitness (np.array): fitness values of all individuals

        Returns:
            np.array: returns offspring
        """

        total_offspring = []

        for i in range(0, self.population_size, 2):

            indv_1 = self.tournament_selection(population, fitness)
            indv_2 = self.tournament_selection(population, fitness)

            offspring = np.zeros(shape=(self.n_offspring, self.n_genomes))

            for child in offspring:
                cross_distribution = np.random.uniform(0, 1)
                child += cross_distribution * indv_1 + (1 - cross_distribution) * indv_2
                child = self.mutate(child)
                total_offspring.append(child)

        return np.asarray(total_offspring)


##############################
##### Load Data
##############################

history = {}

with open(f"./{experiment_name}/simulation_results.json", "r") as file:
    history = json.load(file)

print(2 * "\n")
print(5 * "-" + " Historical Best " + 5 * "-", end="\n\n")
print(history["enemies"][str(env.enemyn)])

##############################
##### Simulation
##############################

hyperp = {
    "n_vars": (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5,
    "population_size": 88,
    "n_offspring": 3,
    "mutation_sigma": 0.5,
    "generations": 14,
    "n_best": 3,
}


algo = GA(n_genomes=hyperp["n_vars"], population_size=hyperp["population_size"], n_offspring=hyperp["n_offspring"], mutation_p=hyperp["mutation_sigma"])


network = {"weights": [], "fitness": -99999999}

# Initialize

population_w = algo.initialize_population()
population_f = evaluate(population_w)

# Evolve

print(2 * "\n")
print(5 * "-" + " Start Evolving " + 5 * "-", end="\n\n")


for generation in range(1, hyperp["generations"] + 1):

    # PARENT SELECTION + MUTATION | Select parrents and create a new generation
    offspring_w = algo.crossover(population_w, population_f)
    offspring_f = evaluate(offspring_w)

    # Combine the old generation with the new generation
    combined_w = np.vstack((population_w, offspring_w))
    combined_f = np.append(population_f, offspring_f)

    # min-max scale the fitness score such that you can use it as probabilities
    normalized_f = np.asarray(list(map(lambda x: norm(x, combined_f), combined_f)))

    # SURVIVOR SELECTION | select the best individuals from the population
    idx_n_best_individuals = np.argpartition(normalized_f, -hyperp["n_best"])[-hyperp["n_best"] :]
    n_best_individuals_w = combined_w[idx_n_best_individuals]
    n_best_individuals_f = combined_f[idx_n_best_individuals]

    # remove those best from the population
    combined_w = np.delete(combined_w, idx_n_best_individuals, axis=0)
    combined_f = np.delete(combined_f, idx_n_best_individuals, axis=0)
    normalized_f = np.delete(normalized_f, idx_n_best_individuals, axis=0)

    # Calculate a survival probability
    survival_prob = normalized_f / np.sum(normalized_f)

    # Select from population
    selection_idx = np.random.choice(combined_w.shape[0], (hyperp["population_size"] - hyperp["n_best"]), p=survival_prob, replace=False)

    # Chose new population
    population_w = np.vstack((combined_w[selection_idx], n_best_individuals_w))
    population_f = np.append(combined_f[selection_idx], n_best_individuals_f)

    best_idx = np.argmax(population_f)
    best_w = population_w[best_idx]
    best_f = population_f[best_idx]

    if network["fitness"] < best_f:
        network["weights"] = best_w
        network["fitness"] = best_f

    print("\t" + f"Generation: {generation:02d}/{hyperp['generations']} - Overall best: {network['fitness']:.2f} | Generation's best: {best_f:.2f} | mean: {np.mean(population_f):.2f}")

print(5 * "-" + " Finished Evolving " + 5 * "-", end="\n\n")

##############################
##### Write to file
##############################

if history["enemies"][str(env.enemyn)]["fitness"] < network["fitness"]:
    history["enemies"][str(env.enemyn)].update({"fitness": network["fitness"].tolist()})
    history["enemies"][str(env.enemyn)].update({"weights": network["weights"].tolist()})
    history["enemies"][str(env.enemyn)].update({"hyper parameters": hyperp})

with open(f"./{experiment_name}/simulation_results.json", "w") as file:
    json.dump(history, file)

##############################
##### Test run
##############################


env.update_parameter("speed", "normal")
env.update_parameter("visuals", "True")
env.play(network["weights"])
