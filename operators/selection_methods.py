import random
import numpy as np

class SelectionMethods:
    def __init__(self):
        pass

    def select_survivor(self, pop, pop_fit, n_pop, selection_type="elitism"):
        """
        Merged method for handling survivor selection based on the selection type.
        """
        population_with_fitness = [(pop[i], pop_fit[i]) for i in range(len(pop))]

        if selection_type == "elitism":
            survivors = self.elitism(population_with_fitness, n_pop)
        elif selection_type == "fitness_proportionate":
            survivors = self.fitness_proportionate_selection(population_with_fitness, n_pop)
        elif selection_type == "tournament":
            survivors = self.tournament_selection(population_with_fitness, n_pop, tournament_size=3)
        elif selection_type == "rank_based":
            survivors = self.rank_based_selection(population_with_fitness, n_pop)
        elif selection_type == "steady_state":
            offspring = []  # For steady-state, you would provide offspring as additional input and needs to be passed accordingly
            survivors = self.steady_state_selection(population_with_fitness, offspring, num_replacements=10)
        elif selection_type == "truncation":
            survivors = self.truncation_selection(population_with_fitness, truncation_threshold=0.5)
        elif selection_type == "random":
            survivors = self.random_replacement(population_with_fitness, num_replacements=10)
        else:
            raise ValueError(f"Unknown selection type: {selection_type}")

        pop_survivors, pop_fit_survivors = zip(*survivors)
        return np.array(pop_survivors), np.array(pop_fit_survivors)


    # Selection operators
    def elitism(self, population, num_survivors):
        """Selects the top 'num_survivors' individuals based on fitness."""
        sorted_population = sorted(population, key=lambda x: x[1], reverse=True)
        return sorted_population[:num_survivors]

    def fitness_proportionate_selection(self, population, num_survivors):
        """Performs fitness proportionate selection (Roulette Wheel)."""
        total_fitness = sum(individual[1] for individual in population)
        selection_probs = [individual[1] / total_fitness for individual in population]
        return random.choices(population, weights=selection_probs, k=num_survivors)

    def tournament_selection(self, population, num_survivors, tournament_size=3):
        """Performs tournament selection."""
        survivors = []
        for _ in range(num_survivors):
            tournament = random.sample(population, tournament_size)
            winner = max(tournament, key=lambda x: x[1])
            survivors.append(winner)
        return survivors
    
    def rank_based_selection(self, population, num_survivors):
        """Performs rank-based selection."""
        sorted_population = sorted(population, key=lambda x: x[1])
        ranks = list(range(1, len(sorted_population) + 1))
        selection_probs = [rank / sum(ranks) for rank in ranks]
        return random.choices(sorted_population, weights=selection_probs, k=num_survivors)

    def steady_state_selection(self, population, offspring, num_replacements):
        """Replaces a small portion of the population with the offspring."""
        sorted_population = sorted(population, key=lambda x: x[1])
        survivors = sorted_population[:-num_replacements]
        return survivors + random.sample(offspring, num_replacements)

    def truncation_selection(self, population, truncation_threshold=0.5):
        """Performs truncation selection, only keeping the top portion."""
        sorted_population = sorted(population, key=lambda x: x[1], reverse=True)
        cutoff = int(len(population) * truncation_threshold)
        return sorted_population[:cutoff]

    def random_replacement(self, population, num_replacements):
        """Performs random replacement of individuals."""
        survivors = random.sample(population, len(population) - num_replacements)
        return survivors
