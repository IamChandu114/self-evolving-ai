import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class SharedPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

class Agent:
    def __init__(self, state_dim, action_dim, lr=0.01):
        self.policy = SharedPolicy(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99

    def select_action(self, state):
        import numpy as np
        if isinstance(state, tuple):
           state = state[0]  # just in case
        state_tensor = torch.from_numpy(np.array(state)).float()
        probs = self.policy(state_tensor)
        action = torch.multinomial(probs, 1).item()
        return action
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            state_tensor = torch.from_numpy(state).float()
            next_tensor = torch.from_numpy(next_state).float()
            target = reward
            if not done:
                target += self.gamma * torch.max(self.policy(next_tensor)).item()
            loss = -torch.log(self.policy(state_tensor)[action]) * target
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()