import random

class SelectionMethods:
    def __init__(self, population):
        self.population = population  # Population is a list of (individual, fitness) tuples

    def elitism(self, num_survivors):
        """Selects the top 'num_survivors' individuals based on fitness."""
        sorted_population = sorted(self.population, key=lambda x: x[1], reverse=True)
        return sorted_population[:num_survivors]

    def fitness_proportionate_selection(self):
        """Performs fitness proportionate selection (Roulette Wheel)."""
        total_fitness = sum(individual[1] for individual in self.population)
        selection_probs = [individual[1] / total_fitness for individual in self.population]
        return random.choices(self.population, weights=selection_probs, k=self.num_survivors)

    def tournament_selection(self, tournament_size=3):
        """Performs tournament selection."""
        survivors = []
        for _ in range(self.num_survivors):
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda x: x[1])
            survivors.append(winner)
        return survivors
    
    def rank_based_selection(self):
        """Performs rank-based selection."""
        sorted_population = sorted(self.population, key=lambda x: x[1])
        ranks = list(range(1, len(sorted_population) + 1))
        selection_probs = [rank / sum(ranks) for rank in ranks]
        return random.choices(sorted_population, weights=selection_probs, k=self.num_survivors)

    def steady_state_selection(self, offspring, num_replacements):
        """Replaces a small portion of the population with the offspring."""
        sorted_population = sorted(self.population, key=lambda x: x[1])
        survivors = sorted_population[:-num_replacements]
        return survivors + random.sample(offspring, num_replacements)

    def truncation_selection(self, truncation_threshold):
        """Performs truncation selection, only keeping the top portion."""
        sorted_population = sorted(self.population, key=lambda x: x[1], reverse=True)
        cutoff = int(len(self.population) * truncation_threshold)
        return sorted_population[:cutoff]

    def random_replacement(self, num_replacements):
        """Performs random replacement of individuals."""
        survivors = random.sample(self.population, len(self.population) - num_replacements)
        return survivors
