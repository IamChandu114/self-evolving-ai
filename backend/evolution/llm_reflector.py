import os
import openai

openai.api_key = os.getenv("AIzaSyDuFg4V3OHQLALszP2hdjYi-DMY_f_yp5Y")  # Set your key in environment

def reflect_performance(task, total_reward, last_rewards):
    """
    Sends performance data to LLM and gets advice.
    """
    avg_reward = sum(last_rewards)/len(last_rewards) if last_rewards else 0
    prompt = f"""
You are an AI coach. The agent is training on {task}.
Current episode reward: {total_reward}.
Average of last 10 episodes: {avg_reward}.

Give a short strategy suggestion for improvement.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        advice = response.choices[0].message.content.strip()
        return advice
    except Exception as e:
        return f"LLM error: {e}"
