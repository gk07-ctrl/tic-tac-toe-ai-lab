# train.py

from tic_tac_toe import TicTacToe
from q_agent import QLearningAgent

def train_agent(episodes=50000, verbose_every=5000):
    env = TicTacToe()
    agent = QLearningAgent(
        alpha=0.2,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.9998
    )

    for episode in range(episodes):
        env.reset()
        state = tuple(env.board)
        done = False

        # Randomly decide if agent is X (1) or O (2) this episode
        agent_player = 1 if episode % 2 == 0 else 2
        opponent_player = 2 if agent_player == 1 else 1

        turn = 1  # X always moves first

        # Track moves for reward assignment
        last_state = state
        last_action = None
        last_player = None

        while not done:
            player = turn
            moves = env.get_legal_moves()

            if player == agent_player:
                action = agent.choose_action(state, moves)
                last_state = state
                last_action = action
                last_player = player
            else:
                # Opponent plays randomly
                action = random.choice(moves)

            next_state, _, done = env.step(action, player)

            if done:
                # Assign reward only for the agent's last move
                if last_player == agent_player:
                    if env.winner == agent_player:
                        reward = 10
                    elif env.winner is None:  # draw
                        reward = 0
                    else:
                        reward = -10
                    next_moves = []
                    agent.update(last_state, last_action, reward, next_state, next_moves, True)
            else:
                # Non-terminal step: no immediate reward, but we can still update
                # For simplicity, we skip non-terminal updates in this basic version.
                pass

            state = next_state
            turn = 2 if turn == 1 else 1

        # Decay epsilon once per episode
        agent.epsilon = max(0.05, agent.epsilon * agent.epsilon_decay)

        if (episode + 1) % verbose_every == 0:
            print(f"Episode {episode + 1}, epsilon={agent.epsilon:.4f}")

    agent.save("q_table.json")
    print("Training done. Q-table saved to q_table.json")
    return agent

if __name__ == "__main__":
    import random
    train_agent(episodes=50000)