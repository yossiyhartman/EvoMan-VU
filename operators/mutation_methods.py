import random
import numpy as np

class MutationMethods:
    def __init__(self):
        pass

    def apply_mutation(self, unmutated_offspring, weight_lower_bound, weight_upper_bound, mutation_sigma, method="complete_genome_mutation"):
        """
        Merged method for creating mutation in offspring based on the mutation type.
        """

        if method == "complete_genome_mutation":
            mutated_offspring = self.complete_genome_mutation(unmutated_offspring, weight_lower_bound, weight_upper_bound, mutation_sigma)
        else:
            raise ValueError(f"Unknown selection type: {method}")

        return mutated_offspring


    # mutation operators
    def complete_genome_mutation(pop, min_value, max_value, sigma):
        mutation = np.random.normal(0, sigma, size=pop.shape)
        new_pop = pop + mutation
        new_pop[new_pop > max_value] = max_value
        new_pop[new_pop < min_value] = min_value
        return new_pop