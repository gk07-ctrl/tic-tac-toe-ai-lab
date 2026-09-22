# q_agent_advanced.py

import json
import os
import random

class QAgentAdvanced:
    def __init__(self, alpha=0.15, gamma=0.95, epsilon=1.0, epsilon_decay=0.9997):
        self.q = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    # --- Symmetry helpers (rotate/flip board) ---
    def _rotate(self, board):
        # 3x3 indices:
        # 0 1 2
        # 3 4 5
        # 6 7 8
        return [
            board[6], board[3], board[0],
            board[7], board[4], board[1],
            board[8], board[5], board[2]
        ]

    def _flip(self, board):
        # horizontal flip
        return [
            board[2], board[1], board[0],
            board[5], board[4], board[3],
            board[8], board[7], board[6]
        ]

    def _symmetries(self, board):
        b = list(board)
        syms = []
        for _ in range(4):
            syms.append(tuple(b))
            syms.append(tuple(self._flip(b)))
            b = self._rotate(b)
        return syms

    def _get_canonical(self, board):
        # choose lexicographically smallest representation as key
        syms = self._symmetries(board)
        return min(syms)

    # --- Q-table access with symmetry ---
    def _get_q_dict(self, board):
        key = self._get_canonical(board)
        if key not in self.q:
            self.q[key] = {a: 0.0 for a in range(9)}
        return self.q[key]

    def choose_action(self, board, moves):
        if random.random() < self.epsilon:
            return random.choice(moves)
        q = self._get_q_dict(board)
        return max(moves, key=lambda a: q.get(a, 0.0))

    def _update_symmetric(self, board, action, delta):
        # update all symmetric equivalents of (board, action)
        syms = self._symmetries(board)
        seen = set()
        for s in syms:
            key = self._get_canonical(s)
            if key in seen:
                continue
            seen.add(key)
            # map action through transforms is complex; for simplicity,
            # we only update the canonical key's action value directly.
            # This is a simplified symmetry: we share state values, not exact action mapping.
            if action in self.q.setdefault(key, {}):
                self.q[key][action] += delta
            else:
                self.q[key] = {a: 0.0 for a in range(9)}
                self.q[key][action] = delta

    def update(self, board, action, reward, next_board, next_moves, done):
        q = self._get_q_dict(board)
        if done:
            target = reward
        else:
            nq = self._get_q_dict(next_board)
            target = reward + self.gamma * max(nq[a] for a in next_moves)
        delta = self.alpha * (target - q[action])
        q[action] += delta

        # Simple symmetric sharing: apply same delta to canonical states
        self._update_symmetric(board, action, delta * 0.5)

        if done:
            self.epsilon = max(0.05, self.epsilon * self.epsilon_decay)

    # --- Reward shaping helpers ---
    def reward_for_move(self, env, player, before_board, after_board, action):
        # Base reward
        if env.done:
            if env.winner == player:
                return 10.0
            elif env.winner is None:
                return 0.0
            else:
                return -10.0

        r = 0.0

        # Center bonus (early game)
        if action == 4 and before_board[4] == 0:
            empties = sum(1 for x in before_board if x == 0)
            if empties >= 7:
                r += 3.0

        # Corner bonus (early)
        corners = [0,2,6,8]
        if action in corners and before_board[action] == 0:
            empties = sum(1 for x in before_board if x == 0)
            if empties >= 6:
                r += 2.0

        # Penalty: allow opponent two-in-a-row threat
        opponent = 3 - player
        for a,b,c in [(0,1,2),(3,4,5),(6,7,8),
                      (0,3,6),(1,4,7),(2,5,8),
                      (0,4,8),(2,4,6)]:
            line = [after_board[a], after_board[b], after_board[c]]
            opp_count = sum(1 for x in line if x == opponent)
            empty_count = sum(1 for x in line if x == 0)
            if opp_count == 2 and empty_count == 1:
                r -= 3.0

        return r

    # --- Save / Load ---
    def save(self, path="q_table_advanced.json"):
        data = {str(k): v for k, v in self.q.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, path="q_table_advanced.json"):
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.q = {tuple(eval(k)): v for k, v in data.items()}
        return True