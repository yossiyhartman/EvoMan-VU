import random
import numpy as np

class ParentSelectionMethods:
    def __init__(self):
        pass

    def select_parents(self, pop, pop_fit, n_parents, smoothing=1, method="fitness_prop_selection"):
        """
        Merged method for creating mutation in offspring based on the mutation method.
        """

        if method == "fitness_prop_selection":
            parents = self.fitness_prop_selection(pop, pop_fit, n_parents, smoothing=1)
        else:
            raise ValueError(f"Unknown selection method: {method}")

        return parents


    # mutation operators
    def fitness_prop_selection(pop, pop_fit, n_parents, smoothing=1):
        fitness = pop_fit + smoothing - np.min(pop_fit)

        # Fitness proportional selection probability
        fps = fitness / np.sum(fitness)

        # make a random selection of indices
        parent_indices = np.random.choice(np.arange(0, pop.shape[0]), (n_parents, 2), p=fps)
        return pop[parent_indices]