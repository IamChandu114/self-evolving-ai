# backend/evolution/engine.py

import os
import pandas as pd

HISTORY_FILE = "logs/evolution_history.csv"


class EvolutionEngine:
    def __init__(self):
        os.makedirs("logs", exist_ok=True)

        # Load history if exists
        if os.path.exists(HISTORY_FILE):
            self.history = pd.read_csv(HISTORY_FILE).to_dict("records")
        else:
            self.history = []

    def record(self, episode: int, task: str, reward: float):
        """
        Save one evolution step
        """
        entry = {
            "episode": episode,
            "task": task,
            "reward": reward
        }
        self.history.append(entry)

        # Persist to disk
        df = pd.DataFrame(self.history)
        df.to_csv(HISTORY_FILE, index=False)

    def get_history(self):
        """
        Used by dashboard
        """
        return self.history
