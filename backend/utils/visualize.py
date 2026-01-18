import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt

def plot_rewards(log_file="logs/rewards.csv"):
    df = pd.read_csv(log_file)
    plt.figure(figsize=(12,6))
    plt.plot(df["Episode"], df["TotalReward"], label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("AI Learning Curve")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_rewards()
