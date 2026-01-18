import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import csv
import os
from collections import deque
from datetime import datetime

# =========================
# Policy Network
# =========================
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# Agent
# =========================
class Agent:
    def __init__(self, state_dim, action_dim, lr=1e-3, epsilon=0.1):
        self.policy = PolicyNet(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = 0.99
        self.epsilon = epsilon
        self.action_dim = action_dim
        self.memory = []

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state = np.array(state, dtype=np.float32)
        state_tensor = torch.from_numpy(state)
        probs = self.policy(state_tensor)
        probs = torch.clamp(probs, 1e-6, 1.0)
        probs = probs / probs.sum()
        return torch.multinomial(probs, 1).item()

    def store(self, transition):
        self.memory.append(transition)

    def update(self):
        if len(self.memory) == 0:
            return

        states, actions, rewards = zip(*self.memory)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        probs = self.policy(states)
        selected_probs = probs[range(len(actions)), actions]
        loss = -torch.mean(torch.log(selected_probs) * rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory = []


# =========================
# Environment Factory
# =========================
def make_env(task):
    if task == "cartpole":
        return gym.make("CartPole-v1")
    elif task == "mountaincar":
        return gym.make("MountainCar-v0")
    else:
        raise ValueError("Unknown task")


# =========================
# Logging function
# =========================
LOG_FILE = "evolution_history.csv"

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Episode", "Task", "Reward"])

def log_reward(episode, task, reward):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), episode, task, reward])


# =========================
# EVOLUTION LOOP
# =========================
if __name__ == "__main__":
    # USER SETTINGS
    tasks = ["cartpole", "mountaincar"]
    num_episodes = 50
    epsilon = 0.1

    # Create agents & envs
    envs = {}
    agents = {}
    for task in tasks:
        env = make_env(task)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        envs[task] = env
        agents[task] = Agent(state_dim, action_dim, epsilon=epsilon)

    episode = 0
    while episode < num_episodes:
        episode += 1
        task = random.choice(tasks)
        env = envs[task]
        agent = agents[task]

        state, _ = env.reset()
        total_reward = 0

        for _ in range(500):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store((state, action, reward))
            total_reward += reward
            state = next_state
            if done:
                break

        agent.update()
        print(f"[TASK: {task}] Episode {episode} | Total Reward: {total_reward}")
        log_reward(episode, task, total_reward)
