# q_agent.py

import json
import os
import random

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.9995):
        self.q = {}  # Q-table: state_tuple -> {action: q_value}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def get_q(self, state):
        state_t = tuple(state)
        if state_t not in self.q:
            self.q[state_t] = {a: 0.0 for a in range(9)}
        return self.q[state_t]

    def choose_action(self, state, moves):
        if random.random() < self.epsilon:
            return random.choice(moves)
        q = self.get_q(state)
        return max(moves, key=lambda a: q.get(a, 0.0))

    def update(self, state, action, reward, next_state, next_moves, done):
        q = self.get_q(state)
        if done:
            target = reward
        else:
            nq = self.get_q(next_state)
            target = reward + self.gamma * max(nq[a] for a in next_moves)
        q[action] += self.alpha * (target - q[action])

        if done:
            self.epsilon = max(0.05, self.epsilon * self.epsilon_decay)

    def save(self, path="q_table.json"):
        data = {str(k): v for k, v in self.q.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, path="q_table.json"):
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.q = {tuple(eval(k)): v for k, v in data.items()}
        return True