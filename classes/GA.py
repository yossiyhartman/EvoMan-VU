import numpy as np


class GA:

    def __init__(self, n_genomes, population_size, n_offspring, mutation_p, elites) -> None:
        self.n_genomes = n_genomes
        self.population_size = population_size
        self.n_offspring = n_offspring
        self.mutation_p = mutation_p
        self.elites = elites

    @classmethod
    def norm(self, x, pfit_pop):
        if (max(pfit_pop) - min(pfit_pop)) > 0:
            x_norm = (x - min(pfit_pop)) / (max(pfit_pop) - min(pfit_pop))
        else:
            x_norm = 0

        if x_norm <= 0:
            x_norm = 0.0000000001
        return x_norm

    ###################
    # INITIALIZING
    ###################

    def initialize_population(self) -> np.array:
        return np.random.normal(size=(self.population_size, self.n_genomes))

    ###################
    # SELECTION
    ###################

    def tournament_selection(self, population: np.array, fitness: np.array) -> np.array:
        selected = []

        for i in range(0, self.population_size, 1):
            index_p1, index_p2 = np.random.randint(0, self.population_size, 2)
            winner = population[index_p1] if fitness[index_p1] > fitness[index_p2] else population[index_p2]
            selected.append(winner)

        return np.asarray(selected)

    def eletist_selection(self, population: np.array, fitness: np.array, top: int = 2) -> np.array:

        idx_n_best_individuals = np.argpartition(fitness, -top)[-top:]
        n_best_individuals_w = population[idx_n_best_individuals]
        n_best_individuals_f = fitness[idx_n_best_individuals]

        population = np.delete(population, idx_n_best_individuals, axis=0)
        fitness = np.delete(fitness, idx_n_best_individuals, axis=0)

        return n_best_individuals_w, n_best_individuals_f, population, fitness

    def survival_selection(self, population: np.array, fitness: np.array) -> np.array:

        normalized_f = np.asarray(list(map(lambda x: self.norm(x, fitness), fitness)))

        # Calculate a survival probability
        survival_prob = normalized_f / np.sum(normalized_f)

        selection_idx = np.random.choice(population.shape[0], size=self.population_size - self.elites, p=survival_prob, replace=False)

        return population[selection_idx], fitness[selection_idx]

    ###################
    # MUTATION
    ###################

    def mutate(self, offspring: np.array) -> np.array:
        for individual in offspring:
            mutation_dist = np.random.uniform(0, 1, size=self.n_genomes) <= 0.5
            individual += mutation_dist * np.random.normal(0, self.mutation_p, size=self.n_genomes)
        return offspring

    ###################
    # CROSSOVER
    ###################

    def crossover(self, parents: np.array) -> np.array:

        total_offspring = []

        for i in range(0, self.population_size, 2):

            offspring = np.zeros(shape=(self.n_offspring, self.n_genomes))

            for child in offspring:
                cross_distribution = np.random.uniform(0, 1)
                child += cross_distribution * parents[i] + (1 - cross_distribution) * parents[i + 1]
                total_offspring.append(child)

        return np.asarray(total_offspring)
