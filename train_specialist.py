import datetime as dt
from tkinter.tix import Tree
from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
from classes.GA import GA
from classes.DataHandler import DataHandler


##############################
##### Initialize Environment
##############################

n_hidden_neurons = 10

env = Environment(
    experiment_name="logs",
    enemies=[3],
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

def norm(x, pfit_pop):
    if (max(pfit_pop) - min(pfit_pop)) > 0:
        x_norm = (x - min(pfit_pop)) / (max(pfit_pop) - min(pfit_pop))
    else:
        x_norm = 0

    if x_norm <= 0:
        x_norm = 0.0000000001
    return x_norm

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
    "population_size": 88,
    "n_offspring": 3,
    "mutation_sigma": 0.35,
    "generations": 15,
    "n_best": 3,
}

##############################
##### Simulation
##############################

algo = GA(n_genomes=hyperp["n_vars"], population_size=hyperp["population_size"], n_offspring=hyperp["n_offspring"], mutation_p=hyperp["mutation_sigma"])

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

    # PARENT SELECTION + MUTATION | Select parrents and create a new generation
    offspring_w = algo.crossover(population_w, population_f)
    offspring_f = evaluate(offspring_w)

    # Combine the old generation with the new generation
    combined_w = np.vstack((population_w, offspring_w))
    combined_f = np.append(population_f, offspring_f)

    # min-max scale the fitness score such that you can use it as probabilities
    normalized_f = np.asarray(list(map(lambda x: norm(x, combined_f), combined_f)))

    # SURVIVOR SELECTION | select the best individuals from the population
    idx_n_best_individuals = np.argpartition(normalized_f, -hyperp["n_best"])[-hyperp["n_best"] :]
    n_best_individuals_w = combined_w[idx_n_best_individuals]
    n_best_individuals_f = combined_f[idx_n_best_individuals]

    # remove those best from the population
    combined_w = np.delete(combined_w, idx_n_best_individuals, axis=0)
    combined_f = np.delete(combined_f, idx_n_best_individuals, axis=0)
    normalized_f = np.delete(normalized_f, idx_n_best_individuals, axis=0)

    # Calculate a survival probability
    survival_prob = normalized_f / np.sum(normalized_f)

    # Select from population
    selection_idx = np.random.choice(combined_w.shape[0], (hyperp["population_size"] - hyperp["n_best"]), p=survival_prob, replace=False)

    # Chose new population
    population_w = np.vstack((combined_w[selection_idx], n_best_individuals_w))
    population_f = np.append(combined_f[selection_idx], n_best_individuals_f)

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

if data_handler.champions[f"enemy {env.enemyn}"]['fitness'] < battle_results['fitness']:
    data_handler.champions[f"enemy {env.enemyn}"]['run'] = run_time
    data_handler.champions[f"enemy {env.enemyn}"]['fitness'] = battle_results['fitness'].tolist()
    data_handler.champions[f'enemy {env.enemyn}']['weights'] = battle_results['weights'].tolist()
    data_handler.champions[f'enemy {env.enemyn}']['battle time'] = t
    data_handler.champions[f'enemy {env.enemyn}']['victorious'] = p > e
    data_handler.champions[f'enemy {env.enemyn}']['hyper parameters'] = hyperp

    data_handler.save_champions()



