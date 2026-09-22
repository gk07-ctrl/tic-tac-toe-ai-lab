# Tic-Tac-Toe AI Lab (Minimax + Q-Learning)

An interactive lab that compares **classical search (Minimax)** and **reinforcement learning (Q-learning)** in Tic-Tac-Toe.  
Runs fully offline on low-end hardware (Celeron CPU, 8 GB RAM) with no GPU or cloud dependencies.

## Quick start

```bash
pip install streamlit
streamlit run app.py
```

Then open the URL shown in terminal (usually http://localhost:8501).

## Features

- **Play tab**:  
  - You vs Unbeatable AI (Minimax).  
  - You vs Strong Q-Agent (Advanced RL with shaped rewards and symmetry).

- **Learning Lab tab**:  
  - Train the Q-agent in chunks.  
  - Evaluate against random or Minimax opponents.  
  - Visualize **win/draw/loss learning curves**.

- **AI Reasoning tab**:  
  - Inspect move-by-move evaluations:
    - Minimax: optimal vs suboptimal moves.
    - Q-agent: Q-values for each legal move.
  - Short interpretations of what the scores mean.

An interactive lab that compares **classical search (Minimax)** and **reinforcement learning (Q-learning)** in Tic-Tac-Toe.  
Runs fully offline on low-end hardware (Celeron CPU, 8 GB RAM) with no GPU or cloud dependencies.

## Features

- **Play tab**:  
  - You vs Unbeatable AI (Minimax).  
  - You vs Strong Q-Agent (Advanced RL with shaped rewards and symmetry).

- **Learning Lab tab**:  
  - Train the Q-agent in chunks.  
  - Evaluate against random or Minimax opponents.  
  - Visualize **win/draw/loss learning curves**.

- **AI Reasoning tab**:  
  - Inspect move-by-move evaluations:
    - Minimax: optimal vs suboptimal moves.
    - Q-agent: Q-values for each legal move.
  - Short interpretations of what the scores mean.

## How it works

### Minimax

Minimax recursively evaluates all possible future moves and chooses the action that maximizes the AI’s outcome assuming optimal play from the opponent. For Tic-Tac-Toe, this guarantees **perfect play**: the game always ends in a draw against a perfect opponent. [14][15]

### Q-Learning (Advanced Agent)

The advanced Q-agent learns a **Q-table** mapping board states to action values using:

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \big(r + \gamma \max_{a'} Q(s',a') - Q(s,a)\big)
\]

- \(s\): board state (with symmetry reduction).  
- \(a\): move.  
- \(r\): shaped reward (win/loss/draw + heuristics).  
- \(\alpha\): learning rate, \(\gamma\): discount factor.

Over thousands of self-play games, the agent learns strong heuristics (control center, block threats) without explicit search.

## Running locally

Requires Python 3.10+ and Streamlit.

```bash
pip install streamlit
streamlit run app.py
```

Then open the shown local URL (usually http://localhost:8501).

## Why this project?

- Demonstrates both **search-based AI** and **reinforcement learning** with **empirical evaluation**.
- Includes **learning curves** and **explainable move reasoning**, not just gameplay.
- Fully **offline** and lightweight; suitable for low-spec machines.