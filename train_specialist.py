import os
import datetime as dt
from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
from classes.GA import GA
from classes.DataHandler import DataHandler

##############################
##### Specify log file destination
##############################

# This is the folder that collects the logs of the runs
log_folder = "EA1"

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

##############################
##### Hyper Parameter Selection
##############################


hyperp = {
    # Number of weights in network
    "network_weights": (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5,
    # Total number of individuals per generation
    "population_size": 88,
    # Number of generations
    "generations": 30,
    # the amount of offspring are created per pair of parents. (e.g. 2 parents create 3 children )
    "n_offspring": 3,
    # Refers to the probability to which an individual is considered for mutation
    "mutation_p_individual": 0.5,
    # Refers to the probability to which an genome is mutated
    "mutation_p_genome": 0.5,
    # Refers to amount of mutation a potential g_nomes can receive
    "mutation_sigma": 0.35,
    # Refers to the number of best individuals which are garuanteed to move to the next generation
    "n_best": 5,
    #
    "exploration.active": False,
    # The amount of generations which are used for exploration
    "exploration.period": 10,
    # The amount of generations which are used to go from explorative to exploitative
    "pivot.period": 10,
    # What are the 'final' values of the paramters
    "endpoints": {
        "mutation_p_individual": 0.2,
        "mutation_p_genome": 0.1,
        "mutation_sigma": 0.1,
        "n_best": 20,
    },
}

parameter_steps = {
    "mutation_p_individual": np.linspace(start=hyperp["mutation_p_individual"], stop=hyperp["endpoints"]["mutation_p_individual"], num=hyperp["pivot.period"]),
    "mutation_p_genome": np.linspace(start=hyperp["mutation_p_genome"], stop=hyperp["endpoints"]["mutation_p_genome"], num=hyperp["pivot.period"]),
    "mutation_sigma": np.linspace(start=hyperp["mutation_sigma"], stop=hyperp["endpoints"]["mutation_sigma"], num=hyperp["pivot.period"]),
    "n_best": np.linspace(start=hyperp["n_best"], stop=hyperp["endpoints"]["n_best"], num=hyperp["pivot.period"], dtype=int),
}

##############################
##### Simulation
##############################

algo = GA(
    n_genomes=hyperp["network_weights"],
    population_size=hyperp["population_size"],
    n_offspring=hyperp["n_offspring"],
    mutation_p_individual=hyperp["mutation_p_individual"],
    mutation_p_genome=hyperp["mutation_p_genome"],
    mutation_sigma=hyperp["mutation_sigma"],
)

# save best individual
battle_results = {"fitness": -99, "weights": []}

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
    "mutation_p_individual": hyperp["mutation_p_individual"],
    "mutation_p_genome": hyperp["mutation_p_genome"],
    "mutation_sigma": hyperp["mutation_sigma"],
    "n_best": hyperp["n_best"],
    "phase": "initialize",
}

# Value indicates how many times the algorithm trains on specific enemy
n_runs = 10

for _ in range(n_runs):

    # Evolve population
    print(2 * "\n" + 7 * "-" + " Start Evolving " + 7 * "-", end="\n\n")
    print("{:<23} {:<6} {:<12} {:<13} {:<14} {:<15} {:<12} {:<12} {:<23} {:<20} {:<15} {:<10} {:<10}".format(*data_handler.logs_headers))

    # Initialize population
    population_w = algo.initialize_population()
    population_f = evaluate(population_w)

    data_log.update(
        {
            "run": dt.datetime.today().strftime("%d/%m/%Y %H:%M:%S"),
            "max.fitness": np.round(np.max(population_f), 3),
            "mean.fitness": np.round(np.mean(population_f), 3),
            "median.fitness": np.round(np.median(population_f), 3),
            "min.fitness": np.round(np.min(population_f), 3),
            "std.fitness": np.round(np.std(population_f), 3),
        }
    )

    data_handler.add_log([v for k, v in data_log.items()])
    print("{:<23} {:<6} {:<12} {:<13} {:<14} {:<15} {:<12} {:<12} {:<23} {:<20} {:<15} {:<10} {:<10}".format(*[v for k, v in data_log.items()]))

    for generation in range(1, hyperp["generations"] + 1):

        if hyperp["exploration.active"]:
            if generation <= hyperp["exploration.period"]:
                phase = "exploring"
            elif hyperp["exploration.period"] < generation <= hyperp["exploration.period"] + hyperp["pivot.period"]:
                phase = "pivoting"
                idx = generation - (hyperp["exploration.period"] + 1)
                hyperp.update(
                    {
                        "mutation_p_individual": parameter_steps["mutation_p_individual"][idx],
                        "mutation_p_genome": parameter_steps["mutation_p_genome"][idx],
                        "mutation_sigma": parameter_steps["mutation_sigma"][idx],
                        "n_best": parameter_steps["n_best"][idx],
                    }
                )
            else:
                phase = "exploiting"
        else:
            phase = "-"

        # PARENT SELECTION
        parents = algo.tournament_selection(population_w, population_f)

        # CROSSOVER
        offspring_w = algo.crossover(parents)

        # MUTATION
        offspring_w = algo.mutate(offspring_w)

        # EVALUATE
        offspring_f = evaluate(offspring_w)

        # COMBINE
        combined_w = np.vstack((population_w, offspring_w))
        combined_f = np.append(population_f, offspring_f)

        # Save the n-best individuals
        n_best_w, n_best_f, combined_w, combined_f = algo.eletist_selection(combined_w, combined_f, hyperp["n_best"])

        # SURVIVAL SELECTION
        selected_w, selected_f = algo.survival_selection(combined_w, combined_f, hyperp["population_size"] - hyperp["n_best"])

        # Set new population
        population_w = np.vstack((selected_w, n_best_w))
        population_f = np.append(selected_f, n_best_f)

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
                "mutation_p_individual": np.round(hyperp["mutation_p_individual"], 3),
                "mutation_p_genome": np.round(hyperp["mutation_p_genome"], 3),
                "mutation_sigma": np.round(hyperp["mutation_sigma"], 3),
                "n_best": hyperp["n_best"],
                "phase": phase,
            }
        )

        data_handler.add_log([v for k, v in data_log.items()])
        print("{:<23} {:<6} {:<12} {:<13} {:<14} {:<15} {:<12} {:<12} {:<23} {:<20} {:<15} {:<10} {:<10}".format(*[v for k, v in data_log.items()]))

    print(2 * "\n" + 7 * "-" + " Finished Evolving " + 7 * "-", end="\n\n")


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
