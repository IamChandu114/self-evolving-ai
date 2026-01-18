import random
from collections import deque

# -------------------------------
# Standard Replay Memory
# -------------------------------
class ReplayMemory:
    def __init__(self, capacity=10000):
        """
        Stores experiences for training the AI.
        Each experience is a tuple: (state, action, reward, next_state, done)
        """
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Return current size of memory
        """
        return len(self.memory)


# -------------------------------
# Prioritized Replay Memory
# -------------------------------
class PrioritizedMemory:
    def __init__(self, capacity=10000):
        """
        Prioritized memory stores experiences with priorities based on reward magnitude.
        Helps AI learn faster by focusing on important experiences.
        """
        self.memory = []
        self.priorities = []
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        """
        Add experience to memory with priority = abs(reward)
        """
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
            self.priorities.pop(0)

        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(abs(reward))  # bigger reward/failure = higher priority

    def sample(self, batch_size):
        """
        Sample experiences based on priority
        """
        total = sum(self.priorities)
        if total == 0:
            # If all priorities are zero, sample randomly
            return random.sample(self.memory, batch_size)

        probs = [p / total for p in self.priorities]
        indices = random.choices(range(len(self.memory)), k=batch_size, weights=probs)
        return [self.memory[i] for i in indices]

    def __len__(self):
        return len(self.memory)
