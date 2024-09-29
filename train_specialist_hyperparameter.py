import logging
import os
import sys
from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
from classes.GA import GA
import optuna

##############################
##### Specify log file destination
##############################

# This is the folder that collects the logs of the runs
log_folder = "output_hyperparameter"

if not os.path.exists(f"./{log_folder}"):
    os.mkdir(f"./{log_folder}")

##############################
##### Initialize Environment
##############################

n_hidden_neurons = 10

env = Environment(
    experiment_name=log_folder,
    enemies=[4],  # Enemies of interest 1, 3, 4
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


##############################
##### Hyperparameter Search Space (Objective)
##############################


def objective(trial):
    # Hyperparameters to tune, sampled using Optuna's trial object
    network_weights = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
    population_size = trial.suggest_int("population_size", 40, 80)  # Tuning population size
    population_size = population_size if (population_size % 2 == 0) else (population_size + 1)
    n_offspring = 2
    generations = trial.suggest_int("generations", 10, 20)  # Tuning generations
    reproduce_p = trial.suggest_float("reproduce_p", 0.1, 0.5)  # Tuning mutation probability per individual
    mutation_p_individual = trial.suggest_float("mutation_p_individual", 0.2, 0.8)  # Tuning mutation probability per individual
    mutation_p_genome = trial.suggest_float("mutation_p_genome", 0.2, 0.8)  # Tuning mutation probability per genome
    mutation_sigma = trial.suggest_float("mutation_sigma", 0.05, 1.0)  # Tuning mutation sigma
    tournament_size = trial.suggest_int("tournament_size", 2, 10)
    n_best = trial.suggest_int("n_best", 1, 20)

    algo = GA(n_genomes=network_weights, population_size=population_size, n_offspring=n_offspring)

    # Initialize population
    population_w = algo.initialize_population()
    population_f = evaluate(population_w)

    for gen in range(1, generations + 1):

        # PARENT SELECTION
        parents_w, parents_f = algo.tournament_selection(population_w, population_f, tournament_size)

        # CROSSOVER
        offspring_w = algo.crossover(parents_w, p=reproduce_p)

        # MUTATION
        offspring_w = algo.mutate(
            offspring=offspring_w,
            p_mutation=mutation_p_individual,
            p_genome=mutation_p_genome,
            sigma_mutation=mutation_sigma,
        )

        # EVALUATE
        offspring_f = evaluate(offspring_w)

        # # COMBINE
        # combined_w = np.vstack((population_w, offspring_w))
        # combined_f = np.append(population_f, offspring_f)

        # SELECTION (Survival/Elitist)
        # if n_best == population_size:
        #     population_w, population_f, _, _ = algo.eletist_selection(parents_w, parents_f, n_best)

        if n_best > 0:
            n_best_w, n_best_f, population_w, population_f = algo.eletist_selection(population_w, population_f, n_best)

            # COMBINE
            combined_w = np.vstack((population_w, offspring_w))
            combined_f = np.append(population_f, offspring_f)

            selected_w, selected_f = algo.survival_selection(combined_w, combined_f, population_size - n_best)

            population_w = np.vstack((selected_w, n_best_w))
            population_f = np.append(selected_f, n_best_f)

        else:

            # COMBINE
            combined_w = np.vstack((population_w, offspring_w))
            combined_f = np.append(population_f, offspring_f)

            population_w, population_f = algo.survival_selection(combined_w, combined_f, population_size)

    # The metric we want to minimize/maximize (let's say we're maximizing the fitness)
    return np.max(population_f)


##############################
##### Set up and run the Optuna Study
##############################

# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

# Set up Optuna with RandomSampler and MedianPruner
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(), pruner=optuna.pruners.MedianPruner())  # Since we are maximizing fitness


# Callback function to log each trial after it's done
def log_trial_result(study, trial):
    print(f"Trial {trial.number} finished.")
    print(f"Value: {trial.value}")
    print(f"Params: {trial.params}")


# Run the optimization
study.optimize(objective, n_trials=25, callbacks=[log_trial_result])


# To access the best trial
print("Best trial:")
trial = study.best_trial

print(f"Fitness: {trial.value}")
print("Hyperparameters: ")
for key, value in trial.params.items():
    print(f"{key}: {value}")
