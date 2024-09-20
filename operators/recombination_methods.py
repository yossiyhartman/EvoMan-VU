import random
import numpy as np

class RecombinationMethods:
    def __init__(self):
        pass

    def create_offspring(self, parents, method="one_point_crossover"):
        """
        Merged method for creating offspring based on the recombination type.
        """

        if method == "one_point_crossover":
            offspring = self.one_point_crossover(parents)
        else:
            raise ValueError(f"Unknown selection type: {method}")

        return offspring


    # Recombination operators
    def one_point_crossover(parents):
        parentsA, parentsB = np.hsplit(parents, 2)
        roll = np.random.uniform(size=parentsA.shape)
        offspring = parentsA * (roll >= 0.5) + parentsB * (roll < 0.5)
        
        # squeeze to get rid of the extra dimension created during parent selecting
        return np.squeeze(offspring, 1)