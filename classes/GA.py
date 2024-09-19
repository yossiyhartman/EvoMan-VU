import numpy as np


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
