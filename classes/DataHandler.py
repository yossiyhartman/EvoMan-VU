import json
import os


class DataHandler:
    def __init__(self) -> None:
        self.logs_headers = ["run", "enemy", "generation", "fitness", "mean.fitness", "std.fitness"]
        self.log = []
        self.champions = {}

    def add_log(self, data):
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

        except FileNotFoundError:
            with open(f"{path}/{file}", "w") as f:
                for _ in range(1, 9):
                    self.champions[f"enemy {_}"] = {
                        "run": 0,
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
