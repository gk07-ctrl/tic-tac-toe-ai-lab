# tic_tac_toe.py

class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [0] * 9
        self.done = False
        self.winner = None

    def get_legal_moves(self):
        return [i for i, v in enumerate(self.board) if v == 0]

    def _check_winner(self):
        wins = [
            (0,1,2), (3,4,5), (6,7,8),
            (0,3,6), (1,4,7), (2,5,8),
            (0,4,8), (2,4,6)
        ]
        b = self.board
        for a,b1,c in wins:
            if b[a] != 0 and b[a] == b[b1] == b[c]:
                return b[a]
        if all(v != 0 for v in b):
            return "draw"
        return None

    def step(self, action, player):
        if self.done or action not in self.get_legal_moves():
            return tuple(self.board), -10, True

        self.board[action] = player
        res = self._check_winner()

        if res == "draw":
            self.done = True
            self.winner = None
            return tuple(self.board), 0, True
        elif res in (1, 2):
            self.done = True
            self.winner = res
            return tuple(self.board), 0, True

        return tuple(self.board), 0, False