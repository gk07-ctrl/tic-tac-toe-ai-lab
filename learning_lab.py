# learning_lab.py

from tic_tac_toe import TicTacToe
from q_agent_advanced import QAgentAdvanced
from minimax_agent import best_move
import random

def evaluate_agent(agent, opponent_type="random", episodes=200):
    """
    Play `episodes` games: agent (X) vs opponent (O).
    opponent_type: "random" or "minimax"
    Returns: wins, draws, losses
    """
    env = TicTacToe()
    wins = draws = losses = 0

    for _ in range(episodes):
        env.reset()
        state = tuple(env.board)
        done = False
        turn = 1  # agent is X

        while not done:
            moves = env.get_legal_moves()
            if turn == 1:
                action = agent.choose_action(list(state), moves)
            else:
                if opponent_type == "random":
                    action = random.choice(moves)
                else:
                    # opponent is minimax as O
                    action = best_move(list(state), 2, 1)

            next_state, _, done = env.step(action, turn)
            state = next_state
            turn = 2 if turn == 1 else 1

        if env.winner == 1:
            wins += 1
        elif env.winner is None:
            draws += 1
        else:
            losses += 1

    return wins, draws, losses

def run_training_experiment(total_episodes=10000, eval_every=1000, eval_episodes=200, opponent_type="random"):
    """
    Train a QAgentAdvanced in chunks, evaluating periodically.
    Returns:
      - history: list of dicts: {"episodes": ..., "wins": ..., "draws": ..., "losses": ...}
      - final_agent: trained agent
    """
    env = TicTacToe()
    agent = QAgentAdvanced(
        alpha=0.15,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.9997
    )

    history = []
    trained = 0

    while trained < total_episodes:
        # Train a chunk
        chunk = eval_every
        for episode in range(chunk):
            env.reset()
            board = list(env.board)
            done = False

            agent_player = 1 if (trained + episode) % 2 == 0 else 2
            turn = 1

            while not done:
                player = turn
                moves = env.get_legal_moves()

                if player == agent_player:
                    action = agent.choose_action(board, moves)
                else:
                    action = random.choice(moves)

                before = board.copy()
                next_state, _, done = env.step(action, player)
                board = list(next_state)

                reward = agent.reward_for_move(env, player, before, board, action)
                next_moves = env.get_legal_moves()
                agent.update(before, action, reward, board, next_moves, done)

                turn = 2 if turn == 1 else 1

            agent.epsilon = max(0.05, agent.epsilon * agent.epsilon_decay)

        trained += chunk

        # Evaluate
        wins, draws, losses = evaluate_agent(agent, opponent_type=opponent_type, episodes=eval_episodes)
        history.append({
            "episodes": trained,
            "wins": wins,
            "draws": draws,
            "losses": losses
        })

    agent.save("q_table_advanced.json")
    return history, agent