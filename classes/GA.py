import numpy as np


class GA:

    def __init__(self, n_genomes, population_size, n_offspring, mutation_p) -> None:
        self.n_genomes = n_genomes
        self.population_size = population_size
        self.n_offspring = n_offspring
        self.mutation_p = mutation_p

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
        return np.random.uniform(size=(self.population_size, self.n_genomes))

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

    def survival_selection(self, population: np.array, fitness: np.array, size: int) -> np.array:

        normalized_f = np.asarray(list(map(lambda x: self.norm(x, fitness), fitness)))

        # Calculate a survival probability
        survival_prob = normalized_f / np.sum(normalized_f)

        selection_idx = np.random.choice(population.shape[0], size=size, p=survival_prob, replace=False)

        return population[selection_idx], fitness[selection_idx]


    def roulette_wheel_selection(self, population: np.array, fitness: np.array) -> np.array:
        min_fitness = np.min(fitness)
        if min_fitness < 0:
            fitness = fitness - min_fitness
        
        total_fitness = np.sum(fitness)
        
        if total_fitness == 0:
            probabilities = np.ones_like(fitness) / len(fitness)
        else:
            probabilities = fitness / total_fitness
        
        indices = np.random.choice(len(population), size=self.population_size, p=probabilities)
        return population[indices]


    def rank_selection(self, population: np.array, fitness: np.array) -> np.array:
        ranks = np.argsort(fitness)
        rank_probabilities = np.arange(1, self.population_size + 1) / np.sum(np.arange(1, self.population_size + 1))
        selected_indices = np.random.choice(ranks, size=self.population_size, p=rank_probabilities)
        return population[selected_indices]


    def stochastic_universal_sampling(self, population: np.array, fitness: np.array) -> np.array:
        total_fitness = np.sum(fitness)
        
        if total_fitness == 0:
            return population  
        
        distance = total_fitness / self.population_size
        start_point = np.random.uniform(0, distance)
        points = [start_point + i * distance for i in range(self.population_size)]

        cumulative_fitness = np.cumsum(fitness)
        selected = []
        i, j = 0, 0
        
        while i < self.population_size and j < len(population):
            if points[i] < cumulative_fitness[j]:
                selected.append(population[j])
                i += 1  
            else:
                j += 1  
                
        return np.asarray(selected)

    
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
