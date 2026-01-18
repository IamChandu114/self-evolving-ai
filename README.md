SELF-EVOLVING AI SYSTEM WITH LIVE REINFORCEMENT LEARNING DASHBOARD

Live Demo:
https://self-evolving-ai-znibnua7kic6skvhy4vh5h.streamlit.app/

OVERVIEW

This project implements a self-evolving reinforcement learning system with a real-time interactive dashboard.
It demonstrates how intelligent agents learn, adapt, and improve through continuous interaction with environments, while their learning progress is visualized live for users.

Unlike notebook-only ML projects, this is a full end-to-end AI system: training engine + live monitoring + cloud deployment.

KEY FEATURES

• Reinforcement Learning Engine

Policy-based RL agent implemented in PyTorch

Supports multiple environments (CartPole, MountainCar)

Continuous self-evolution loop

Reward tracking per episode

• Live Interactive Dashboard

Built using Streamlit

Real-time reward graphs

Training history visualization

Live training status

• User Controls

Select task/environment

Adjust exploration rate (epsilon)

Control number of training episodes

Start evolution from the UI

• Cloud Deployment

Deployed on Streamlit Cloud

Publicly accessible demo

No local setup required for viewers

PROJECT STRUCTURE

self-evolving-ai/
|
|-- backend/
| |-- evolution/
| | |-- evolution_loop.py (RL training & self-evolution)
| |
| |-- ui/
| |-- dashboard.py (Live Streamlit dashboard)
|
|-- requirements.txt
|-- README.md
|-- .gitignore

TECH STACK

Programming Language: Python
Reinforcement Learning: PyTorch
Environments: Gymnasium
Dashboard: Streamlit
Visualization: Matplotlib
Data Handling: NumPy, Pandas
Deployment: Streamlit Cloud

HOW IT WORKS

The RL agent interacts with the environment.

Rewards are collected for each episode.

The policy network updates using policy gradients.

Training statistics are logged continuously.

The Streamlit dashboard reads and visualizes live data.

Users can modify parameters while observing learning behavior.

RUN LOCALLY

Clone the repository
git clone https://github.com/IamChandu114/self-evolving-ai.git

cd self-evolving-ai

Create and activate virtual environment
python -m venv venv
venv\Scripts\activate (Windows)

Install dependencies
pip install -r requirements.txt

Run evolution loop
python backend/evolution/evolution_loop.py

Launch dashboard
streamlit run backend/ui/dashboard.py

REAL-WORLD APPLICATIONS

• Robotics and autonomous control
• Traffic signal optimization
• Logistics and delivery systems
• Game AI
• Energy optimization systems
• Adaptive decision-making platforms

WHAT RECRUITERS SEE IN THIS PROJECT

• End-to-end AI system ownership
• Strong reinforcement learning fundamentals
• Real-time monitoring and visualization
• Debugging and production thinking
• Cloud deployment experience
• Practical, non-toy AI engineering

FUTURE ENHANCEMENTS

• Multi-agent evolution
• Advanced RL algorithms (PPO, A3C)
• Persistent database logging
• Model checkpointing
• Dockerized deployment
• Experiment comparison dashboard

AUTHOR

Chandu Vemula
Aspiring AI / ML Engineer

GitHub: https://github.com/IamChandu114

FINAL NOTE

This project represents learning by building real systems, not just training models.

If you find this project useful or interesting, feel free to star the repository.
