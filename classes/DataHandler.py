import json
import os


class DataHandler:
    def __init__(self) -> None:
        self.logs_headers = [
            "run",
            "enemy",
            "generation",
            "max.fitness",
            "mean.fitness",
            "median.fitness",
            "min.fitness",
            "std.fitness",
            "tournament.size",
            "mutation.p.individual",
            "mutation.p.genome",
            "mutation.sigma",
            "population.size",
            "elites",
            "phase",
        ]
        self.log = []
        self.champions = {}

    @classmethod
    def print_data(self, data):
        print("{:<23} {:<6} {:<12} {:<13} {:<14} {:<15} {:<12} {:<12} {:<17} {:<23} {:<20} {:<15} {:<17} {:<10} {:<10}".format(*data))

    def add_log(self, data, print=True):
        if print:
            self.print_data(data)
        self.log.append(data)

    def save_logs(self, path="./logs", file="logs.txt"):

        headers = False

        if not os.path.exists(f"{path}/{file}"):
            headers = True

        with open(f"{path}/{file}", "a+") as f:
            if headers:
                f.write(",".join(self.logs_headers) + "\n")

            for entry in self.log:
                f.write(",".join(str(x) for x in entry) + "\n")

        self.log = []

    def load_champions(self, path="./logs", file="champions.json"):
        try:
            with open(f"{path}/{file}", "r") as f:
                self.champions = json.load(f)
                return self.champions

        except FileNotFoundError:
            with open(f"{path}/{file}", "w") as f:
                for _ in range(1, 9):
                    self.champions[f"enemy {_}"] = {
                        "fitness": -99999,
                        "victorious": False,  # did the champion beat the enemy
                        "battle time": -99,  #
                        "weights": [],
                        "hyper parameters": {
                            "n_vars": -99,
                            "population_size": -99,
                            "n_offspring": -99,
                            "mutation_sigma": 0,
                            "generations": -99,
                            "n_best": -99,
                        },
                    }
                json.dump(self.champions, f, ensure_ascii=False, indent=4)

    def save_champions(self, path="./logs", file="champions.json"):
        with open(f"{path}/{file}", "w") as f:
            json.dump(self.champions, f, ensure_ascii=False, indent=4)
