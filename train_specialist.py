import os
import datetime as dt
from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
from classes.GA import GA
from classes.DataHandler import DataHandler




log_folder = "selection - tournament - logs"

if not os.path.exists(f'./{log_folder}'):
    os.mkdir(f'./{log_folder}')


##############################
##### Initialize Environment
##############################

n_hidden_neurons = 10

env = Environment(
    experiment_name=log_folder,
    enemies=[1],
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
data_handler.load_champions()

print("\n" + 7 * "-" + " Current Champion " + 7 * "-", end="\n\n")
print(f"\tFitness:\t{data_handler.champions[f"enemy {env.enemyn}"]['fitness']}")
print(f"\tVictorious:\t{data_handler.champions[f"enemy {env.enemyn}"]['victorious']}")
print(f"\tbattle time:\t{data_handler.champions[f"enemy {env.enemyn}"]['battle time']}")

##############################
##### Hyper Parameter Selection
##############################


hyperp = {
    "n_vars": (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5,
    "population_size": 44,
    "n_offspring": 3,
    "mutation_sigma": 0.35,
    "generations": 10,
    "n_best": 3,
}

##############################
##### Simulation
##############################

algo = GA(n_genomes=hyperp["n_vars"],
          population_size=hyperp["population_size"],
          n_offspring=hyperp["n_offspring"],
          mutation_p=hyperp["mutation_sigma"],
          elites=hyperp['n_best'])

# This values acts as an id for a specific run
run_time = dt.datetime.today().strftime("%d/%m/%Y %H:%M:%S")

# save best individual
battle_results = {
    'fitness' : -99,
    'weights': []
}

# Evolve population
print(2 * "\n" + 7 * "-" + " Start Evolving " + 7 * "-", end="\n\n")
print("\t{:<20} {:<8} {:<12} {:<10} {:<15} {:<12}".format(*data_handler.logs_headers))

# Initialize population
population_w = algo.initialize_population()
population_f = evaluate(population_w)

data = [run_time, env.enemyn, 0, np.round(np.max(population_f),3), np.round(np.mean(population_f),3), np.round(np.std(population_f),3)]
data_handler.add_log(data)
print("\t{:<20} {:<8} {:<12} {:<10} {:<15} {:<12}".format(*data))

for generation in range(1, hyperp["generations"] + 1):

    # PARENT SELECTION
    parents = algo.tournament_selection(population_w, population_f)

    # CROSSOVER
    offspring_w = algo.crossover(parents)

    # MUTATION
    offspring_w = algo.mutate(offspring_w)
    offspring_f = evaluate(offspring_w)

    # COMBINE
    combined_w = np.vstack((population_w, offspring_w))
    combined_f = np.append(population_f, offspring_f)

    # ELITIST SELECTION
    # n_best_w, n_best_f, combined_w, combined_f = algo.eletist_selection(combined_w, combined_f, hyperp['n_best'])

    # SURVIVAL SELECTION
    population_w, population_f = algo.survival_selection(combined_w, combined_f)

    # population_w = np.vstack((selected_w, n_best_w))
    # population_f = np.append(selected_f, n_best_f)
    
    best_idx = np.argmax(population_f)
    best_w = population_w[best_idx]
    best_f = population_f[best_idx]

    if battle_results["fitness"] < best_f:
        battle_results["weights"] = best_w
        battle_results["fitness"] = best_f


    data = [run_time, env.enemyn, generation, np.round(np.max(population_f),3), np.round(np.mean(population_f),3), np.round(np.std(population_f),3)]
    data_handler.add_log(data)
    print("\t{:<20} {:<8} {:<12} {:<10} {:<15} {:<12}".format(*data))
    

print(2 * "\n" + 7 * "-" + " Finished Evolving " + 7 * "-", end="\n\n")



# ##############################
# ##### Write to file (logs)
# ##############################

save_logs = False

if save_logs:
    data_handler.save_logs()


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

save_champion = False

if save_champion and (data_handler.champions[f"enemy {env.enemyn}"]['fitness'] < battle_results['fitness']):
    data_handler.champions[f"enemy {env.enemyn}"]['run'] = run_time
    data_handler.champions[f"enemy {env.enemyn}"]['fitness'] = battle_results['fitness'].tolist()
    data_handler.champions[f'enemy {env.enemyn}']['weights'] = battle_results['weights'].tolist()
    data_handler.champions[f'enemy {env.enemyn}']['battle time'] = t
    data_handler.champions[f'enemy {env.enemyn}']['victorious'] = p > e
    data_handler.champions[f'enemy {env.enemyn}']['hyper parameters'] = hyperp

    data_handler.save_champions()



