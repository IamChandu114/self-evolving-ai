# backend/main.py

import os
import csv
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from agents.cartpole_agent import PolicyNetwork
from utils.memory import ReplayMemory

reward_log_file = "logs/rewards.csv"

# Only write header if file does not exist
if not os.path.exists(reward_log_file) or os.path.getsize(reward_log_file) == 0:
    with open(reward_log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "TotalReward"])

# Hyperparameters
# -------------------
episodes = 1000           # Number of episodes per run (continuous loop will override)
batch_size = 64
gamma = 0.99              # Discount factor
learning_rate = 0.01
checkpoint_interval = 50

# -------------------
# Environment Setup
# -------------------
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# -------------------
# Initialize AI Core
# -------------------
policy = PolicyNetwork(state_dim, action_dim)
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

# -------------------
# Memory for Experience Replay
# -------------------
memory = ReplayMemory(capacity=10000)

# -------------------
# Logging Setup
# -------------------
os.makedirs("logs/checkpoints", exist_ok=True)
reward_log_file = "logs/rewards.csv"
if not os.path.exists(reward_log_file):
    with open(reward_log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "TotalReward"])

# -------------------
# Helper Functions
# -------------------
def select_action(state):
    state = torch.from_numpy(state).float()
    probs = policy(state)
    action = torch.multinomial(probs, 1).item()
    return action

def save_checkpoint(episode):
    path = f"logs/checkpoints/policy_episode_{episode}.pt"
    torch.save(policy.state_dict(), path)
    print(f"Checkpoint saved: {path}")

def load_best_checkpoint():
    checkpoints = os.listdir("logs/checkpoints")
    if not checkpoints:
        return
    latest = sorted(checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
    policy.load_state_dict(torch.load(f"logs/checkpoints/{latest}"))
    print(f"Loaded checkpoint: {latest}")

# -------------------
# Training Loop
# -------------------
def continuous_train():
    episode = 1
    load_best_checkpoint()
    
    while True:  # Infinite loop for self-evolution
        state, _ = env.reset()  # ignore info
        done = False
        total_reward = 0

        while not done:
            action = select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Store experience
            memory.push(state, action, reward, next_state, done)
            state = next_state

            # Train on batch from memory
            if len(memory) >= batch_size:
                batch = memory.sample(batch_size)
                for s, a, r, s_next, d in batch:
                    s_tensor = torch.from_numpy(s).float()
                    s_next_tensor = torch.from_numpy(s_next).float()
                    target = r
                    if not d:
                        target += gamma * torch.max(policy(s_next_tensor)).item()
                    loss = -torch.log(policy(s_tensor)[a]) * target
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        # Log rewards
        with open(reward_log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, total_reward])

        # Checkpoint
        if episode % checkpoint_interval == 0:
            save_checkpoint(episode)

        print(f"Episode {episode}, Total Reward: {total_reward}")
        episode += 1

# -------------------
# Entry Point
# -------------------
if __name__ == "__main__":
    continuous_train()
