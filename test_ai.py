from tic_tac_toe import TicTacToe
from q_agent import QLearningAgent
import random

env = TicTacToe()
agent = QLearningAgent()
loaded = agent.load("q_table.json")
if not loaded:
    print("No q_table.json found. Run 'python train.py' first.")
    exit()

env.reset()
state = tuple(env.board)
done = False
turn = 1  # 1 = AI (X), 2 = random (O)

while not done:
    if turn == 1:
        moves = env.get_legal_moves()
        action = agent.choose_action(state, moves)
        print("AI (X) chooses:", action)
    else:
        moves = env.get_legal_moves()
        action = random.choice(moves)
        print("Random (O) chooses:", action)

    next_state, _, done = env.step(action, turn)

    b = env.board
    symbols = ["." if x == 0 else ("X" if x == 1 else "O") for x in b]
    print(
        f"{symbols[0]} {symbols[1]} {symbols[2]}\n"
        f"{symbols[3]} {symbols[4]} {symbols[5]}\n"
        f"{symbols[6]} {symbols[7]} {symbols[8]}\n"
    )

    state = next_state
    turn = 2 if turn == 1 else 1

print("Winner:", env.winner)  # 1 = AI, 2 = random, None = draw
