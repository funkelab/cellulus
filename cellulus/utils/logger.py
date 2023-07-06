from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


class Logger:
    def __init__(self, keys: List[str], title: str):
        self.keys = keys
        self.title = title
        self.window = plt.subplots()
        self.data: Dict[str, List[float]] = {k: [] for k in keys}
        print(f"Created logger with keys: {keys}")

    def add(self, key, value):
        assert key in self.data, "Key not in data"
        self.data[key].append(value)

    def write(self):
        for key in self.data:
            self.data[key]

        df = pd.DataFrame.from_dict(self.data)
        df.to_csv(self.title + ".csv")

    def plot(self):
        fig, ax = self.window
        ax.cla()
        for key in self.data:
            data = self.data[key]
            ax.plot(range(len(data)), data, marker=".")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(self.title)
        plt.draw()
        plt.savefig(self.title + ".png")
