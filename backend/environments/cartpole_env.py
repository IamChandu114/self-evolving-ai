import gymnasium as gym
import numpy as np

def make_env():
    env = gym.make("CartPole-v1")

    original_reset = env.reset
    def reset():
        state, info = original_reset()
        return np.array(state, dtype=np.float32)
    env.reset = reset

    original_step = env.step
    def step(action):
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        return np.array(next_state, dtype=np.float32), reward, terminated, truncated, info
    env.step = step

    return env
