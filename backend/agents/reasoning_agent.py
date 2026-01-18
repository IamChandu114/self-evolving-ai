import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")  # Set your API key

class ReasoningAgent:
    def __init__(self):
        self.memory = []

    def store_experience(self, task, episode, reward):
        self.memory.append({
            "task": task,
            "episode": episode,
            "reward": reward
        })
        if len(self.memory) > 50:
            self.memory.pop(0)

    def analyze_and_recommend(self):
        prompt = "You are an AI coach. Analyze these experiences and recommend improvements:\n"
        for exp in self.memory:
            prompt += f"Task: {exp['task']}, Episode: {exp['episode']}, Reward: {exp['reward']}\n"
        prompt += "\nRecommend next learning strategy."
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            return f"LLM Error: {e}"
