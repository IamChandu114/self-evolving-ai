import random

TASKS = ["cartpole", "mountaincar"]
task_performance = {task: [] for task in TASKS}

def update_task_performance(task, reward):
    task_performance[task].append(reward)
    if len(task_performance[task]) > 100:
        task_performance[task] = task_performance[task][-100:]

def select_task():
    avg_rewards = {task: (sum(rewards)/len(rewards) if rewards else 0)
                   for task, rewards in task_performance.items()}
    # Choose the task with lowest average reward
    worst_task = min(avg_rewards, key=avg_rewards.get)
    return worst_task

