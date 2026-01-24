import streamlit as st
import pandas as pd
import subprocess
import threading
import time
import os

LOG_FILE = "evolution_history.csv"

st.set_page_config(page_title="Self-Evolving AI Dashboard", layout="wide")
st.title("🧠 Self-Evolving AI Dashboard")

tasks = st.multiselect(
    "Select Tasks",
    ["cartpole", "mountaincar"],
    default=["cartpole", "mountaincar"]
)

epsilon = st.slider("Exploration (epsilon)", 0.0, 1.0, 0.1, 0.01)
num_episodes = st.number_input(
    "Number of episodes",
    min_value=10,
    max_value=1000,
    value=50,
    step=10
)

run_button = st.button("Start Evolution Loop")

def start_evolution():
    cmd = ["python", "backend/evolution/evolution_loop.py"]
    env_vars = os.environ.copy()
    env_vars["TASKS"] = ",".join(tasks)
    env_vars["EPSILON"] = str(epsilon)
    env_vars["NUM_EPISODES"] = str(num_episodes)
    subprocess.Popen(cmd, env=env_vars)

if run_button:
    st.info("Evolution loop started...")

# 🔁 Auto-refresh every second (SAFE)
st.subheader("📈 Reward History")
chart_placeholder = st.empty()

if os.path.exists(LOG_FILE):
    df = pd.read_csv(LOG_FILE)
    if not df.empty:
        df = df[df["Task"].isin(tasks)]
        chart_placeholder.line_chart(
            df.pivot(index="Episode", columns="Task", values="Reward")
        )

st.experimental_rerun()

