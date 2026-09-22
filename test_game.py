from tic_tac_toe import TicTacToe
import random

env = TicTacToe()
env.reset()

players = [1, 2]
turn = 0

while not env.done:
    player = players[turn % 2]
    moves = env.get_legal_moves()
    action = random.choice(moves)
    next_state, reward, done = env.step(action, player)
    print("Board:", next_state, "Winner:", env.winner, "Done:", done)
    turn += 1

print("Game over. Winner:", env.winner) 