import os
import datetime as dt
import pprint

from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
from classes.GA import GA
from classes.DataHandler import DataHandler
import pandas as pd

##############################
##### Specify log file destination
##############################

# This is the folder that collects the logs of the runs
log_folder = "output_EA1"

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
##### Load Data
##############################

data_handler = DataHandler()
data_handler.load_champions(path=log_folder)

print("\n" + 7 * "-" + " Current Champion " + 7 * "-", end="\n\n")
print(f"\tFitness:\t{data_handler.champions[f'enemy {env.enemyn}']['fitness']}")
print(f"\tVictorious:\t{data_handler.champions[f'enemy {env.enemyn}']['victorious']}")
print(f"\tbattle time:\t{data_handler.champions[f'enemy {env.enemyn}']['battle time']}")


results = {}

enemies = [4, 6, 8]


##############################
##### Hyper Parameter Selection
##############################


def set_hyperparameters():
    return {
        "network_weights": (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5,
        "population_size": 70,
        "n_offspring": 2,
        "generations": 30,
        "reproduce_p": 0.47419626805893167,
        "mutation_p_individual": 0.28753745118457064,
        "mutation_p_genome": 0.7369551652097696,
        "mutation_sigma": 0.8613221369404007,
        "tournament_size": 8,
        "n_best": 2,
    }


hyperp = set_hyperparameters()

##############################
##### Simulation
##############################

algo = GA(n_genomes=hyperp["network_weights"], population_size=hyperp["population_size"], n_offspring=hyperp["n_offspring"])

# save best individual

# Value indicates how many times the algorithm trains on specific enemy
n_runs = 10

for _ in range(n_runs):

    battle_results = {"fitness": -99, "weights": []}

    env.update_parameter("enemies", [4])

    # Evolve population
    print(2 * "\n" + 7 * "-" + f" run {_}, parameter reset" + 7 * "-", end="\n\n")
    pprint.pprint(hyperp)

    # data log
    data_log = {
        "run": None,
        "enemy": env.enemyn,
        "generation": 0,
        "max.fitness": None,
        "mean.fitness": None,
        "median.fitness": None,
        "min.fitness": None,
        "std.fitness": None,
        "tournament_size": hyperp["tournament_size"],
        "mutation_p_individual": hyperp["mutation_p_individual"],
        "mutation_p_genome": hyperp["mutation_p_genome"],
        "mutation_sigma": hyperp["mutation_sigma"],
        "population_size": hyperp["population_size"],
        "n_best": hyperp["n_best"],
        "phase": "-",
    }

    # Evolve population
    print(2 * "\n" + 7 * "-" + " Start Evolving " + 7 * "-", end="\n\n")
    data_handler.print_data(data_handler.logs_headers)

    # Initialize population
    population_w = algo.initialize_population()
    population_f = evaluate(population_w)

    data_log.update(
        {
            "run": dt.datetime.today().strftime("%d/%m/%Y %H:%M:%S"),
            "generation": 0,
            "max.fitness": np.round(np.max(population_f), 3),
            "mean.fitness": np.round(np.mean(population_f), 3),
            "median.fitness": np.round(np.median(population_f), 3),
            "min.fitness": np.round(np.min(population_f), 3),
            "std.fitness": np.round(np.std(population_f), 3),
            "tournament_size": np.round(hyperp["tournament_size"], 3),
            "mutation_p_individual": np.round(hyperp["mutation_p_individual"], 3),
            "mutation_p_genome": np.round(hyperp["mutation_p_genome"], 3),
            "mutation_sigma": np.round(hyperp["mutation_sigma"], 3),
            "n_best": hyperp["n_best"],
        }
    )

    data_handler.add_log([v for k, v in data_log.items()])

    for generation in range(1, hyperp["generations"] + 1):

        # PARENT SELECTION
        parents_w, parents_f = algo.tournament_selection(population_w, population_f, hyperp["tournament_size"])

        # CROSSOVER
        offspring_w = algo.crossover(parents_w, p=hyperp["reproduce_p"])

        # MUTATION
        offspring_w = algo.mutate(offspring=offspring_w, p_mutation=hyperp["mutation_p_individual"], p_genome=hyperp["mutation_p_genome"], sigma_mutation=hyperp["mutation_sigma"])

        # EVALUATE
        offspring_f = evaluate(offspring_w)

        if hyperp["n_best"] > 0:
            n_best_w, n_best_f, population_w, population_f = algo.eletist_selection(population_w, population_f, hyperp["n_best"])

            # COMBINE
            combined_w = np.vstack((population_w, offspring_w))
            combined_f = np.append(population_f, offspring_f)

            selected_w, selected_f = algo.survival_selection(combined_w, combined_f, hyperp["population_size"] - hyperp["n_best"])

            population_w = np.vstack((selected_w, n_best_w))
            population_f = np.append(selected_f, n_best_f)

        else:
            # COMBINE
            combined_w = np.vstack((population_w, offspring_w))
            combined_f = np.append(population_f, offspring_f)

            population_w, population_f = algo.survival_selection(combined_w, combined_f, hyperp["population_size"] - hyperp["n_best"])

        best_idx = np.argmax(population_f)
        best_w = population_w[best_idx]
        best_f = population_f[best_idx]

        if battle_results["fitness"] < best_f:
            battle_results["weights"] = best_w
            battle_results["fitness"] = best_f

        data_log.update(
            {
                "generation": generation,
                "max.fitness": np.round(np.max(population_f), 3),
                "mean.fitness": np.round(np.mean(population_f), 3),
                "median.fitness": np.round(np.median(population_f), 3),
                "min.fitness": np.round(np.min(population_f), 3),
                "std.fitness": np.round(np.std(population_f), 3),
                "tournament_size": np.round(hyperp["tournament_size"], 3),
                "mutation_p_individual": np.round(hyperp["mutation_p_individual"], 3),
                "mutation_p_genome": np.round(hyperp["mutation_p_genome"], 3),
                "mutation_sigma": np.round(hyperp["mutation_sigma"], 3),
                "n_best": hyperp["n_best"],
            }
        )

        data_handler.add_log([v for k, v in data_log.items()])

    print(2 * "\n" + 7 * "-" + " Finished Evolving " + 7 * "-", end="\n\n")

    hyperp = set_hyperparameters()

    for enemy in enemies:
        env.update_parameter("enemies", [enemy])

        f, p, e, t = env.play(battle_results["weights"])

        if not results.get(enemy):
            results[enemy] = []

        results[enemy].append(f)


df = pd.DataFrame(results)
df.to_csv("battle_output.csv")
print(df)

# ##############################
# ##### Write to file (logs)
# ##############################

save_logs = True

if save_logs:
    data_handler.save_logs(path=log_folder)


# ##############################
# ##### Test run
# ##############################

show_test_run = True

if show_test_run:
    env.update_parameter("speed", "normal")
    env.update_parameter("visuals", "True")
    f, p, e, t = env.play(battle_results["weights"])

# ##############################
# ##### (Possibly) Update Champion
# ##############################

save_champion = True

if save_champion and (data_handler.champions[f"enemy {env.enemyn}"]["fitness"] < battle_results["fitness"]):
    data_handler.champions[f"enemy {env.enemyn}"]["fitness"] = battle_results["fitness"].tolist()
    data_handler.champions[f"enemy {env.enemyn}"]["weights"] = battle_results["weights"].tolist()
    data_handler.champions[f"enemy {env.enemyn}"]["battle time"] = t
    data_handler.champions[f"enemy {env.enemyn}"]["victorious"] = p > e
    data_handler.champions[f"enemy {env.enemyn}"]["hyper parameters"] = hyperp

    data_handler.save_champions(path=log_folder)
